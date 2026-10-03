"""Read installed verl sources/AST; never execute a policy update or loss.

Recognize a narrow, evidenced dataflow, not a flag-name heuristic. Unsupported
implementations are undetermined. This is diagnostic-only, not a verl patch.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path


def expression(node):
    return ast.unparse(node) if node is not None else None


def same(node, text):
    return ast.dump(node, include_attributes=False) == ast.dump(ast.parse(text, mode="eval").body, include_attributes=False)


def find_node(text, qualified_name):
    nodes = ast.parse(text).body
    result = None
    for name in qualified_name.split("."):
        found = [n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
        if len(found) != 1:
            raise ValueError(f"source symbol missing/ambiguous: {qualified_name}")
        result = found[0]
        nodes = result.body
    return result


def assignments(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


def calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Name) and n.func.id == name)
                 or (isinstance(n.func, ast.Attribute) and n.func.attr == name))]


def execution_nodes(node, flags):
    """Prune ONLY observed runtime booleans; never assume arbitrary branches."""
    yield node
    if isinstance(node, ast.If) and expression(node.test) in flags:
        for child in node.body if flags[expression(node.test)] else node.orelse:
            yield from execution_nodes(child, flags)
    else:
        for child in ast.iter_child_nodes(node):
            yield from execution_nodes(child, flags)


def evidence(path, text, node, label):
    snippet = "\n".join(text.splitlines()[node.lineno-1:node.end_lineno])
    return dict(symbol=label, source_file=str(path), line_start=node.lineno,
                line_end=node.end_lineno, source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                snippet_sha256=hashlib.sha256(snippet.encode()).hexdigest(),
                # Full methods can be long; branch evidence remains inspectable.
                snippet=snippet if len(snippet) < 5000 else None)


def analyze_sources(*, actor_source, actor_class, loss_source, loss_name,
                    config_source, config_class, worker_source="", trainer_source="", correction_source="", execution_flags=None):
    """CPU probe of exact selected actor branch -> loss arguments -> exp ratio."""
    result = dict(use_rollout_log_probs_symbol_found=False, when_true="undetermined",
        when_false="undetermined", denominator_source="undetermined",
        actor_recompute_path_exists="undetermined", ppo_semantic_acceptability="undetermined",
        facts={}, interpretation="No complete installed-source dataflow proof.", evidence_nodes=[])
    try:
        update = find_node(actor_source, actor_class + ".update_policy")
        compute = find_node(actor_source, actor_class + ".compute_log_prob")
        forward = find_node(actor_source, actor_class + "._forward_micro_batch")
        config = find_node(config_source, config_class)
        loss = find_node(loss_source, loss_name)
        flag_defs = [n for n in config.body if isinstance(n, ast.AnnAssign)
                     and isinstance(n.target, ast.Name) and n.target.id == "use_rollout_log_probs"]
        flags = [n for n in ast.walk(update) if isinstance(n, ast.If)
                 and same(n.test, 'hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs')]
        result["use_rollout_log_probs_symbol_found"] = bool(flag_defs and flags)
        result["facts"]["config_default"] = expression(flag_defs[0].value) if len(flag_defs) == 1 else None
        if len(flags) != 1 or len(flag_defs) != 1:
            return result
        branch = flags[0]
        # Reject a changed branch, competing assignment, or different denominator.
        old = assignments(update, "old_log_prob")
        nested = branch.orelse[0] if len(branch.orelse) == 1 else None
        true = assignments(ast.Module(body=branch.body, type_ignores=[]), "old_log_prob")
        conditional = isinstance(nested, ast.If) and same(nested.test, "on_policy")
        yes = assignments(ast.Module(body=nested.body, type_ignores=[]), "old_log_prob") if conditional else []
        no = assignments(ast.Module(body=nested.orelse, type_ignores=[]), "old_log_prob") if conditional else []
        prefix = [n for n in old if n not in true + yes + no]
        valid_prefix = not prefix or (len(prefix) == 1 and prefix[0].lineno < branch.lineno
                                      and same(prefix[0].value, 'model_inputs["old_log_probs"]'))
        proven_branch = (len(old) in (3, 4) and valid_prefix and len(true) == len(yes) == len(no) == 1
            and same(true[0].value, 'model_inputs["old_log_probs"]')
            and same(yes[0].value, "log_prob.detach()")
            and same(no[0].value, 'model_inputs["old_log_probs"]'))
        on_policy = assignments(update, "on_policy")
        proven_condition = (len(on_policy) == 1
            and same(on_policy[0].value, "len(mini_batches) == 1 and self.config.ppo_epochs == 1"))
        current = [n for n in ast.walk(update) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Tuple) and any(isinstance(v, ast.Name) and v.id == "log_prob" for v in t.elts)
                           for t in n.targets) and isinstance(n.value, ast.Call)
                   and expression(n.value.func) == "self._forward_micro_batch"]
        loss_calls = calls(update, "policy_loss_fn")
        proven_args = (len(loss_calls) == 1 and
            {k.arg: expression(k.value) for k in loss_calls[0].keywords}.get("old_log_prob") == "old_log_prob" and
            {k.arg: expression(k.value) for k in loss_calls[0].keywords}.get("log_prob") == "log_prob")
        factory = assignments(update, "policy_loss_fn")
        mode = assignments(update, "loss_mode")
        proven_factory = (len(factory) == len(mode) == 1 and
            same(factory[0].value, "get_policy_loss_fn(loss_mode)") and
            same(mode[0].value, 'self.config.policy_loss.get("loss_mode", "vanilla")'))
        sub = assignments(loss, "negative_approx_kl")
        ratio = assignments(loss, "ratio")
        proven_ratio = (len(sub) in (1, 2) and same(sub[0].value, "log_prob - old_log_prob")
            and (len(sub) == 1 or same(sub[1].value, "torch.clamp(negative_approx_kl, min=-20.0, max=20.0)"))
            and len(ratio) == 1 and same(ratio[0].value, "torch.exp(negative_approx_kl)"))
        # Exact method identities, independent-forward and temperature evidence.
        eval_call = any(expression(c.func) == "self.actor_module.eval" for c in calls(compute, "eval"))
        no_grad = any(expression(c.func) == "torch.no_grad" for c in calls(compute, "no_grad"))
        forward_calls = calls(compute, "_forward_micro_batch")
        returns = [n for n in ast.walk(compute) if isinstance(n, ast.Return)]
        return_contract = len(returns) == 1 and same(returns[0].value, "(log_probs, entropys)")
        selected = list(execution_nodes(forward, execution_flags or {
            "self.use_remove_padding": False, "self.use_fused_kernels": False}))
        temperature_ops = [n for n in selected if isinstance(n, ast.Call)
                           and expression(n.func) == "logits.div_" and len(n.args) == 1 and same(n.args[0], "temperature")]
        autocast = [expression(c) for c in calls(forward, "autocast")]
        result["facts"].update(branch_proven=proven_branch, on_policy_condition_proven=proven_condition,
            current_forward_proven=len(current) == 1, loss_arguments_proven=proven_args,
            loss_factory_proven=proven_factory, ratio_formula_proven=proven_ratio,
            compute_eval=eval_call, compute_no_grad=no_grad, compute_forward_calls=len(forward_calls),
            compute_returns_logprobs_first=return_contract,
            temperature_division_count=len(temperature_ops), autocast=autocast,
            execution_flags=execution_flags,
            rollout_log_probs_accesses=[expression(n) for n in ast.walk(update) if
                (isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value == "rollout_log_probs") or
                (isinstance(n, ast.Call) and expression(n.func) == "model_inputs.get" and n.args and same(n.args[0], '"rollout_log_probs"'))])
        result["evidence_nodes"] = [("actor", actor_class + ".update_policy.flag_branch", branch),
            ("actor", actor_class + ".update_policy.on_policy", on_policy[0]),
            ("actor", actor_class + ".compute_log_prob", compute),
            ("actor", actor_class + "._forward_micro_batch", forward),
            ("config", config_class, config), ("loss", loss_name, loss)]
        if not all((proven_branch, proven_condition, len(current) == 1, proven_args, proven_factory,
                    proven_ratio, eval_call, no_grad, return_contract, len(forward_calls) == 1, len(temperature_ops) == 1, bool(autocast))):
            return result
        result.update(when_true='model_inputs["old_log_probs"]; origin determined by caller, not flag name',
            when_false='on_policy (one mini-batch AND ppo_epochs==1): current log_prob.detach(); otherwise model_inputs["old_log_probs"]',
            denominator_source='old_log_prob keyword to resolved vanilla loss; exp(clamp(log_prob - old_log_prob, -20,20))',
            ppo_semantic_acceptability="not_supported_by_verl_implementation",
            interpretation="Candidate fix cannot be justified by the assumed unconditional false-flag semantics. The on-policy branch uses the SAME update forward detached, not an independent pre-update O. Trainer recomputation is a separate data-carrier path, not implied by this flag.")
        # Audit separate trainer + worker data carrier, without importing Ray.
        if worker_source and trainer_source:
            worker = find_node(worker_source, "ActorRolloutRefWorker.compute_log_prob")
            trainer = find_node(trainer_source, "RayPPOTrainer.fit")
            worker_calls = calls(worker, "compute_log_prob")
            worker_producers = [n for n in ast.walk(worker) if isinstance(n, ast.Assign) and
                isinstance(n.value, ast.Call) and expression(n.value.func) == "self.actor.compute_log_prob" and
                any(isinstance(t, ast.Tuple) and t.elts and expression(t.elts[0]) == "log_probs" for t in n.targets)]
            payloads = [c for c in calls(worker, "from_dict") if expression(c.func) == "DataProto.from_dict" and any(
                k.arg == "tensors" and isinstance(k.value, ast.Dict) and
                any(isinstance(key, ast.Constant) and key.value == "old_log_probs" and expression(v) == "log_probs"
                    for key, v in zip(k.value.keys, k.value.values)) for k in c.keywords)]
            producer = assignments(trainer, "old_log_prob")
            actor_producer = [n for n in producer if isinstance(n.value, ast.Call)
                              and expression(n.value.func) == "self.actor_rollout_wg.compute_log_prob"]
            carrier = [c for c in calls(trainer, "union") if expression(c.func) == "batch.union"
                       and len(c.args) == 1 and same(c.args[0], "old_log_prob")]
            union_assignments = [n for n in assignments(trainer, "batch") if n.value in carrier]
            bypass_definition = assignments(trainer, "bypass_recomputing_logprobs")
            bypass = [n for n in ast.walk(trainer) if isinstance(n, ast.If) and
                      ((isinstance(n.test, ast.BoolOp) and "bypass_mode" in expression(n.test)) or
                       (same(n.test, "bypass_recomputing_logprobs") and len(bypass_definition) == 1
                        and same(bypass_definition[0].value, 'rollout_corr_config and rollout_corr_config.get("bypass_mode", False)')))
                      and actor_producer and actor_producer[0] in list(ast.walk(ast.Module(body=n.orelse, type_ignores=[])))]
            recompute = (any(expression(c.func) == "self.actor.compute_log_prob" for c in worker_calls)
                         and len(worker_producers) == len(payloads) == 1 and len(actor_producer) == 1
                         and len(carrier) == len(union_assignments) == 1 and len(bypass) == 1)
            result["actor_recompute_path_exists"] = True if recompute else "undetermined"
            result["facts"]["trainer_recompute_carrier"] = (
                "non-bypass trainer: worker compute_log_prob -> DataProto(old_log_probs=log_probs) -> batch.union(old_log_prob); independent of actor flag"
                if recompute else "undetermined")
            result["facts"]["trainer_bypass_condition"] = expression(bypass[0].test) if bypass else None
            result["facts"]["trainer_bypass_definition"] = expression(bypass_definition[0].value) if bypass_definition else None
            result["evidence_nodes"] += [("worker", "ActorRolloutRefWorker.compute_log_prob", worker),
                                         ("trainer", "RayPPOTrainer.fit.old_log_prob", actor_producer[0])] if actor_producer else []
            if correction_source:
                correction = find_node(correction_source, "apply_rollout_correction")
                carriers = [n for n in ast.walk(correction) if isinstance(n, ast.Assign) and
                            any(same(ast.parse(expression(t), mode="eval").body, 'batch.batch["old_log_probs"]') for t in n.targets)]
                proven_rollout = (len(carriers) == 1 and same(carriers[0].value, 'batch.batch["rollout_log_probs"]')
                                  and len(bypass) == 1 and bool(calls(bypass[0], "apply_rollout_correction")))
                result["facts"]["bypass_rollout_carrier"] = (
                    'trainer bypass: apply_rollout_correction assigns batch.old_log_probs = batch.rollout_log_probs'
                    if proven_rollout else "undetermined")
                result["evidence_nodes"].append(("correction", "apply_rollout_correction", correction))
    except (SyntaxError, ValueError, IndexError, AttributeError) as exc:
        result["probe_error"] = str(exc)
        result["ppo_semantic_acceptability"] = "undetermined"
    return result


def inspect_verl_old_logprob_semantics(actor, *, version):
    """Rank-zero runtime probe. Actual installed class, config and loss factory."""
    import verl
    package = Path(inspect.getfile(verl)).resolve().parent
    functions = {name: inspect.unwrap(getattr(type(actor), name)) for name in ("compute_log_prob", "update_policy", "_forward_micro_batch")}
    actor_path = Path(inspect.getsourcefile(type(actor)))
    config_path = Path(inspect.getsourcefile(type(actor.config)))
    factory = inspect.getmodule(type(actor)).__dict__.get("get_policy_loss_fn")
    mode = actor.config.policy_loss.get("loss_mode", "vanilla")
    sources = {"actor": (actor_path, actor_path.read_text(encoding="utf-8")),
               "config": (config_path, config_path.read_text(encoding="utf-8"))}
    runtime = dict(use_rollout_log_probs=actor.config.use_rollout_log_probs,
                   ppo_epochs=actor.config.ppo_epochs, ppo_mini_batch_size=actor.config.ppo_mini_batch_size,
                   policy_loss_mode=mode, compute_method=functions["compute_log_prob"].__qualname__,
                   update_method=functions["update_policy"].__qualname__)
    result = dict(verl_version=version, runtime_gate_config=runtime,
                  ppo_semantic_acceptability="undetermined", use_rollout_log_probs_symbol_found=False)
    try:
        if version != "0.6.1" or mode != "vanilla" or factory is None:
            raise ValueError("unsupported version/loss/factory")
        loss_fn = factory(mode)  # read-only registry lookup, NEVER execute loss
        loss_path = Path(inspect.getsourcefile(loss_fn))
        sources["loss"] = (loss_path, loss_path.read_text(encoding="utf-8"))
        # Discover definitions in actual package, not guessed external filenames.
        for label, symbol in (("worker", "ActorRolloutRefWorker"), ("trainer", "RayPPOTrainer")):
            found = []
            for path in package.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                if symbol not in text:
                    continue
                try:
                    find_node(text, symbol)
                    # The FSDP actor's worker is selected by its actual FSDP
                    # utility imports, not a guessed filename or Megatron peer.
                    fsdp_import = any(isinstance(n, ast.ImportFrom) and n.module == "verl.utils.fsdp_utils"
                                      for n in ast.walk(ast.parse(text)))
                    if label != "worker" or fsdp_import:
                        found.append((path, text))
                except (ValueError, SyntaxError):
                    pass
            if len(found) == 1:
                sources[label] = found[0]
        if "trainer" in sources:
            imports = [n for n in ast.walk(ast.parse(sources["trainer"][1])) if isinstance(n, ast.ImportFrom)
                       and any(a.name == "apply_rollout_correction" for a in n.names) and n.module and n.module.startswith("verl.")]
            modules = {n.module for n in imports}
            if len(modules) == 1:
                path = package.joinpath(*next(iter(modules)).split(".")[1:]).with_suffix(".py")
                sources["correction"] = (path, path.read_text(encoding="utf-8"))
        result.update(analyze_sources(actor_source=sources["actor"][1], actor_class=type(actor).__name__,
            config_source=sources["config"][1], config_class=type(actor.config).__name__,
            loss_source=sources["loss"][1], loss_name=loss_fn.__name__,
            worker_source=sources.get("worker", (None, ""))[1], trainer_source=sources.get("trainer", (None, ""))[1],
            correction_source=sources.get("correction", (None, ""))[1],
            execution_flags={"self.use_remove_padding": actor.use_remove_padding,
                             "self.use_fused_kernels": actor.use_fused_kernels}))
        result["resolved_loss_function"] = loss_fn.__module__ + "." + loss_fn.__qualname__
    except Exception as exc:
        result.update(ppo_semantic_acceptability="undetermined", probe_error=str(exc))
    nodes = result.pop("evidence_nodes", [])
    result["evidence"] = [evidence(*sources[label], node, name) for label, name, node in nodes]
    result["source_files"] = {label: str(path.resolve()) for label, (path, _) in sources.items()}
    result["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in sources.values()}
    result["definition_locations"] = [dict(path=str(path), line=n.lineno, symbol=expression(n))
        for path, text in sources.values() for n in ast.walk(ast.parse(text))
        if isinstance(n, (ast.Attribute, ast.AnnAssign)) and
        ((isinstance(n, ast.Attribute) and n.attr == "use_rollout_log_probs") or
         (isinstance(n, ast.AnnAssign) and expression(n.target) == "use_rollout_log_probs"))]
    return result
