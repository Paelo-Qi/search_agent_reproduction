# SearchVL-SFT-36K tool contract audit

This is a read-only statistical audit. It does not modify trajectories or implement tools.

## 1. Dataset summary

- Dataset: `OpenSearch-VL/Search-VL-SFT-36K`
- Revision: `2c1c460af4fa15bd63210cbf426a96664b959944`
- Trajectories: 36,592
- Tool calls: 96,888
- With tools: 36,186
- Without tools: 406
- Malformed trajectories: 0
- Average / median / max calls per trajectory: 2.648 / 2.0 / 7

## 2. Tool frequency and trajectory usage

| Tool | Calls | Trajectories | Usage | Avg calls when used | Max in one trajectory |
|---|---:|---:|---:|---:|---:|
| `text_search` | 55,258 | 34,823 | 95.17% | 1.587 | 7 |
| `image_search` | 33,161 | 32,838 | 89.74% | 1.010 | 3 |
| `crop` | 4,906 | 3,744 | 10.23% | 1.310 | 2 |
| `layout_parsing` | 2,808 | 2,562 | 7.00% | 1.096 | 3 |
| `super_resolution` | 511 | 511 | 1.40% | 1.000 | 1 |
| `sharpen` | 170 | 170 | 0.46% | 1.000 | 1 |
| `web_search` | 62 | 60 | 0.16% | 1.033 | 2 |
| `perspective_correct` | 12 | 12 | 0.03% | 1.000 | 1 |

## 3. Per-tool argument and observation contracts

### `text_search`

Priority: **high priority to preserve**. Runtime backend may be replaceable if the interface is preserved.

Argument schemas:

- 50,397 (91.20%): `{"encoding": "native", "keys": {"hl": "string", "q": "string", "top_k": "number"}, "type": "object"}`
  - Example: `{"q": "Attabad Lake outflow where does it drain to river", "hl": "en", "top_k": 5}`
- 4,858 (8.79%): `{"encoding": "native", "keys": {"q": "string", "top_k": "number"}, "type": "object"}`
  - Example: `{"q": "broaching machining process internal shapes keyway spline hexagon", "top_k": 5}`
- 3 (0.01%): `{"encoding": "native", "keys": {"hl": "string", "q": "string"}, "type": "object"}`
  - Example: `{"q": "雲のように風のように anime DVD Like the Clouds Like the Wind", "hl": "en"}`

Observation format:

- Types: `{"string": 55258}`
- Text/JSON: 100.00% / 0.00%
- Length min/mean/median/max: 94 / 10944.9 / 11240.0 / 23821
- Top-level JSON keys: `{}`
- Text markers: `{"Title:": 54743, "URL:": 54743, "Content:": 2062, "Search results": 221, "Snippet:": 49}`
- success_like / failure_like / unknown: 46674 / 8584 / 0

success_like examples:

- `fvqa:0` args=`{"q": "Attabad Lake outflow where does it drain to river", "hl": "en", "top_k": 5}`; observation: '<observation>\nTool execution result:\n\n\n============================================================[Passage 1]\nTitle: Attabad Lake - Wikipedia\nURL: https://en.wikipedia.org/wiki/Attabad_Lake\nSummary:\n<think>\nOkay, the user is asking where the outflow of Attabad Lake drains into a river. Let me check the provided webpage content.\n\nLooking at the "Primary outflows" section, it says the outflow is the Gojal River overflowing a landslide dam, with a discharge rate and date. The Hunza River is mentio'
- `fvqa:1` args=`{"q": "2025 economic outlook report Asia released organization", "hl": "en", "top_k": 5}`; observation: '<observation>\nTool execution result:\n\n\n============================================================[Passage 1]\nTitle: Economic Forecasts: Asian Development Outlook December 2025\nURL: https://www.adb.org/outlook/editions/december-2025\nSummary:\n<think>\nOkay, let\'s see. The user is asking for a concise summary related to the "2025 economic outlook report Asia released organization." The webpage title is "Economic Forecasts: Asian Development Outlook December 2025." The content is just "Just a momen'
- `fvqa:2` args=`{"q": "stećak medieval Bosnian tombstone monument history", "hl": "en", "top_k": 5}`; observation: '<observation>\nTool execution result:\n\n\n============================================================[Passage 1]\nTitle: Stećci Medieval Tombstone Graveyards\nURL: https://whc.unesco.org/en/list/1504/\nSummary:\n<think>\nOkay, let\'s tackle this query. The user wants a concise summary about the history of stećak medieval Bosnian tombstone monuments based on the provided webpage content. First, I need to parse the given content.\n\nThe webpage title is "Stećci Medieval Tombstone Graveyards - UNESCO World H'

failure_like examples:

- `fvqa:7` args=`{"q": "Arizona small plane collision crash fatalities deaths", "hl": "en", "top_k": 5}`; observation: "<observation>\nTool execution result:\n\n\n============================================================[Passage 1]\nTitle: Four people died in an Arizona helicopter crash Friday after the ...\nURL: https://www.facebook.com/yahoonews/posts/four-people-died-in-an-arizona-helicopter-crash-friday-after-the-aircraft-appear/1257838849535131/\nSummary:\n<think>\nOkay, let's tackle this query. The user wants a concise summary about the Arizona small plane collision crash fatalities. The webpage content mentions "
- `fvqa:11` args=`{"q": "Casa de la Villa Madrid official opening year history construction", "hl": "en", "top_k": 5}`; observation: "<observation>\nTool execution result:\n\n\n============================================================[Passage 1]\nTitle: Casa de la Villa - Madrid rocks!\nURL: https://madridrocks.iessapereaude.com/our-buildings/casa-de-la-villa/\nSummary:\n<think>\nOkay, let's tackle this query. The user is asking about the official opening year, history, and construction of Casa de la Villa in Madrid. First, I need to extract the relevant info from the provided webpage content.\n\nLooking at the historical context sect"
- `fvqa:27` args=`{"q": "Kinkaku-ji Golden Pavilion Kyoto architectural style", "hl": "en", "top_k": 5}`; observation: "<observation>\nTool execution result:\n\n\n============================================================[Passage 1]\nTitle: ABOUT | Kinkaku-ji | 臨済宗相国寺派\nURL: https://www.shokoku-ji.jp/en/kinkakuji/about/\nSummary:\n<think>\nOkay, let's tackle this query. The user wants a concise summary about the architectural style of Kinkaku-ji (Golden Pavilion) in Kyoto. The provided webpage content is a bit sparse, but there are some key points.\n\nFirst, the title mentions it's part of the Rinzai-sect Zen temple of Sh"

### `image_search`

Priority: **high priority to preserve**. Runtime backend may be replaceable if the interface is preserved.

Argument schemas:

- 33,161 (100.00%): `{"encoding": "native", "keys": {"url": "string"}, "type": "object"}`
  - Example: `{"url": "img_1"}`

Observation format:

- Types: `{"string": 33161}`
- Text/JSON: 100.00% / 0.00%
- Length min/mean/median/max: 54 / 2636.9 / 3387 / 5314
- Top-level JSON keys: `{}`
- Text markers: `{"Search results": 20095, "Title:": 3951, "URL:": 857}`
- success_like / failure_like / unknown: 30959 / 2202 / 0

success_like examples:

- `fvqa:0` args=`{"url": "img_1"}`; observation: '<observation>\nTool execution result:\n{\n  "summary": "<think>\\nOkay, let\'s see. The user wants me to process image search results and extract only the relevant \\"title\\" and \\"source\\" information. They provided a JSON structure with an array of results. My task is to go through each entry in the \\"organic\\" array and pick out the title and source, removing any extra stuff.\\n\\nFirst, I need to check each object in the organic list. For each one, look at the \\"title\\" and \\"source\\" fields. The us'
- `fvqa:2` args=`{"url": "img_1"}`; observation: '<observation>\nTool execution result:\n[{\'title\': "THE GOTHS IN BOSNIA: In the footsteps of the Bosniaks\' letter to Hitler", \'source\': \'Amazon.com\'}, {\'title\': \'Medieval Bosnian stecak tombstones and their history\', \'source\': \'Facebook\'}, {\'title\': \'Who were Bogomils\', \'source\': \'OoCities\'}, {\'title\': \'Bogumil tombstone\', \'source\': \'OSZK\'}, {\'title\': \'File:Radimlja necropolis.jpg - Wikimedia Commons\', \'source\': \'Wikimedia.org\'}, {\'title\': \'Slavic homeland and ethnogenesis | Page 13 | Eupedia Forum'
- `fvqa:8` args=`{"url": "img_1"}`; observation: '<observation>\nTool execution result:\n{\n  "summary": "<think>\\nOkay, let\'s see. The user wants me to extract only the title and source from each of the image search results provided. The input is a JSON object with an \\"organic\\" array containing several entries. Each entry has imageUrl, link, source, thumbnailUrl, and title.\\n\\nFirst, I need to go through each item in the organic array. For each one, I should create a new object that includes just the \\"title\\" and \\"source\\" fields. I have to m'

failure_like examples:

- `fvqa:3` args=`{"url": "img_1"}`; observation: '<observation>\nTool execution result:\n{\n  "summary": "<think>\\nOkay, let\'s see. The user wants me to extract and summarize only the relevant \\"title\\" and \\"source\\" information from the image search results provided. They specified to remove all irrelevant information and keep only the essential identification details.\\n\\nFirst, I need to look through the JSON structure given. The main key is \\"organic\\" which contains an array of results. Each result has fields like imageUrl, link, source, thum'
- `fvqa:5` args=`{"url": "img_1"}`; observation: '<observation>\nTool execution result:\n{\n  "summary": "<think>\\nOkay, let\'s tackle this problem. The user wants me to extract and summarize only the \\"title\\" and \\"source\\" information from the given image search results. They specified to remove all irrelevant info and keep only the essential details.\\n\\nFirst, I\'ll look at the JSON structure provided. There\'s an \\"organic\\" array with several entries. Each entry has \\"title\\" and \\"source\\" fields, which are exactly what\'s needed. The other fie'
- `fvqa:18` args=`{"url": "img_1"}`; observation: '<observation>\nTool execution result:\n{\n  "summary": "<think>\\nOkay, let\'s see. The user wants me to extract and summarize only the relevant \\"title\\" and \\"source\\" information from the image search results. They mentioned removing all irrelevant info and keeping only the essential details.\\n\\nFirst, I need to look through each entry in the \\"organic\\" array. Each entry has \\"title\\", \\"source\\", and other fields like \\"link\\" and \\"imageUrl\\". The task is to create a JSON list with objects that'

### `crop`

Priority: **high priority to preserve**. Runtime backend may be replaceable if the interface is preserved.

Argument schemas:

- 4,906 (100.00%): `{"encoding": "native", "keys": {"height": "number", "image": "string", "width": "number", "x": "number", "y": "number"}, "type": "object"}`
  - Example: `{"image": "img_1", "x": 0, "y": 200, "width": 180, "height": 150}`

Observation format:

- Types: `{"string": 4906}`
- Text/JSON: 100.00% / 0.00%
- Length min/mean/median/max: 311 / 313.8 / 314.0 / 315
- Top-level JSON keys: `{}`
- Text markers: `{}`
- success_like / failure_like / unknown: 4906 / 0 / 0

success_like examples:

- `fvqa:38` args=`{"image": "img_1", "x": 0, "y": 200, "width": 180, "height": 150}`; observation: '<image><observation>\nImage cropped successfully. New image ID: img_3. The cropped image has been uploaded to: http://tj-zw1-303793872-gtzrcx9os6cdhzuhp8egg-1258344706.cos.ap-zhongwei-taiji.myqcloud.com/shawncschen/2026-01-04/shawncschen_fvqa_2026-01-04_fvqa/fvqa_train_2674_trajectory_turn0_crop.png\n</observation>'
- `fvqa:144` args=`{"image": "img_1", "x": 195, "y": 145, "width": 320, "height": 180}`; observation: '<image><observation>\nImage cropped successfully. New image ID: img_3. The cropped image has been uploaded to: http://tj-zw1-303793872-gtzrcx9os6cdhzuhp8egg-1258344706.cos.ap-zhongwei-taiji.myqcloud.com/shawncschen/2026-01-04/shawncschen_fvqa_2026-01-04_fvqa/fvqa_train_2970_trajectory_turn0_crop.png\n</observation>'
- `fvqa:219` args=`{"image": "img_1", "x": 0, "y": 0, "width": 640, "height": 400}`; observation: '<image><observation>\nImage cropped successfully. New image ID: img_3. The cropped image has been uploaded to: http://tj-zw1-303793872-gtzrcx9os6cdhzuhp8egg-1258344706.cos.ap-zhongwei-taiji.myqcloud.com/shawncschen/2026-01-04/shawncschen_fvqa_2026-01-04_fvqa/fvqa_train_2896_trajectory_turn1_crop.png\n</observation>'

### `layout_parsing`

Priority: **medium priority**. Runtime backend may be replaceable if the interface is preserved.

Argument schemas:

- 2,770 (98.65%): `{"encoding": "native", "keys": {"image": "string"}, "type": "object"}`
  - Example: `{"image": "img_1"}`
- 34 (1.21%): `{"encoding": "native", "keys": {"image": "string", "use_chart_recognition": "boolean"}, "type": "object"}`
  - Example: `{"image": "img_1", "use_chart_recognition": true}`
- 4 (0.14%): `{"encoding": "native", "keys": {"image": "string", "use_doc_orientation_classify": "boolean"}, "type": "object"}`
  - Example: `{"image": "img_1", "use_doc_orientation_classify": true}`

Observation format:

- Types: `{"string": 2808}`
- Text/JSON: 100.00% / 0.00%
- Length min/mean/median/max: 87 / 775.0 / 329.0 / 77101
- Top-level JSON keys: `{}`
- Text markers: `{"Content:": 942, "URL:": 3, "Title:": 1}`
- success_like / failure_like / unknown: 2619 / 189 / 0

success_like examples:

- `fvqa:6` args=`{"image": "img_1"}`; observation: '<observation>\n✅ Layout Parsing SUCCESS: Text detected successfully!\nTotal text blocks detected: 2\n\n[Text Block 1]\n  Content: "LASMCER"\n\n[Text Block 2]\n  Content: "20 - 21 Nov 2025 | Toronto, Canada"\n\n============================================================\nALL RECOGNIZED TEXT (Use this for your answer):\n============================================================\nBlock 1: "LASMCER"\nBlock 2: "20 - 21 Nov 2025 | Toronto, Canada"\n\nCombined text:\nLASMCER\n20 - 21 Nov 2025 | Toronto, Canada\n======'
- `fvqa:144` args=`{"image": "img_3"}`; observation: '<observation>\n✅ Layout Parsing SUCCESS: Text detected successfully!\nTotal text blocks detected: 3\n\n[Text Block 1]\n  Content: "HAUHONAHH HAPK\nNATIONAL PARK K"\n\n[Text Block 2]\n  Content: "Природно стани поточне настрем - Salmo truta"\n\n[Text Block 3]\n  Content: "Natural habitat of the trout - Salmo tru"\n\n============================================================\nALL RECOGNIZED TEXT (Use this for your answer):\n============================================================\nBlock 1: "HAUHONAHH HAPK\nNA'
- `fvqa:210` args=`{"image": "img_1"}`; observation: '<observation>\n✅ Layout Parsing SUCCESS: Text detected successfully!\nTotal text blocks detected: 15\n\n[Text Block 1]\n  Content: "THE"\n\n[Text Block 2]\n  Content: "HETWEN"\n\n[Text Block 3]\n  Content: "AND\nGERMANY,"\n\n[Text Block 4]\n  Content: "The Protocol annexed thereto, the Agreement respecting the military occupation of the territories of the Rhine,"\n\n[Text Block 5]\n  Content: "AND THE"\n\n[Text Block 6]\n  Content: "TREATY"\n\n[Text Block 7]\n  Content: "BETWEEN"\n\n[Text Block 8]\n  Content: "FRANCE AND '

failure_like examples:

- `fvqa:1039` args=`{"image": "img_1"}`; observation: '<observation>\n✅ Layout Parsing SUCCESS: Text detected successfully!\nTotal text blocks detected: 27\n\n[Text Block 1]\n  Content: "BACKGROUND\n*↓↓ PRODUCTION of URINE"\n\n[Text Block 2]\n  Content: "SIGNS & SYMPTOMS"\n\n[Text Block 3]\n  Content: "* URINE OUTPUT < 500mL per DAY or 0.5mL/kg/HR"\n\n[Text Block 4]\n  Content: "$ ^{*} $  SHOCK:"\n\n[Text Block 5]\n  Content: "~ TACHYCARDIA, HYPOTENSION, ↓↓ SKIN TURGOR, COOL EXTREMITIES"\n\n[Text Block 6]\n  Content: "* OBSTRUCTIVE URETERAL KIDNEY STONES:"\n\n[Text Block '
- `livevqa:174` args=`{"image": "img_1"}`; observation: "<observation>\nTool execution error:\nLayout parsing error: HTTPSConnectionPool(host='ccs9da36y3a6yaqa.aistudio-app.com', port=443): Read timed out. (read timeout=60)\n</observation>"
- `livevqa:188` args=`{"image": "img_1"}`; observation: '<observation>\nTool execution error:\nAPI request failed with status 500: {"logId":"089e1501-9274-4b60-bf2d-9c6b70c09d7b","errorCode":500,"errorMsg":"Internal server error"}\n</observation>'

### `super_resolution`

Priority: **medium priority**. Runtime backend may be replaceable if the interface is preserved.

Argument schemas:

- 511 (100.00%): `{"encoding": "native", "keys": {"image": "string", "scale": "number"}, "type": "object"}`
  - Example: `{"image": "img_1", "scale": 4}`

Observation format:

- Types: `{"string": 511}`
- Text/JSON: 100.00% / 0.00%
- Length min/mean/median/max: 350 / 351.8 / 352 / 353
- Top-level JSON keys: `{}`
- Text markers: `{}`
- success_like / failure_like / unknown: 511 / 0 / 0

success_like examples:

- `fvqa:1106` args=`{"image": "img_1", "scale": 4}`; observation: '<image><observation>\nSuper resolution enhancement completed successfully. New image ID: img_3. The enhanced image has been uploaded to: http://tj-zw1-303793872-gtzrcx9os6cdhzuhp8egg-1258344706.cos.ap-zhongwei-taiji.myqcloud.com/shawncschen/2026-01-03/shawncschen_fvqa_2026-01-03_fvqa/fvqa_train_2206_trajectory_turn0_super_resolution.png\n</observation>'
- `fvqa:2229` args=`{"image": "img_1", "scale": 4}`; observation: '<image><observation>\nSuper resolution enhancement completed successfully. New image ID: img_3. The enhanced image has been uploaded to: http://tj-zw1-303793872-gtzrcx9os6cdhzuhp8egg-1258344706.cos.ap-zhongwei-taiji.myqcloud.com/shawncschen/2026-01-01/shawncschen_fvqa_2026-01-01_fvqa/fvqa_train_533_trajectory_turn1_super_resolution.png\n</observation>'
- `fvqa:3371` args=`{"image": "img_1", "scale": 4}`; observation: '<image><observation>\nSuper resolution enhancement completed successfully. New image ID: img_3. The enhanced image has been uploaded to: http://tj-zw1-303793872-gtzrcx9os6cdhzuhp8egg-1258344706.cos.ap-zhongwei-taiji.myqcloud.com/shawncschen/2026-01-02/shawncschen_fvqa_2026-01-02_fvqa/fvqa_train_1200_trajectory_turn2_super_resolution.png\n</observation>'

### `sharpen`

Priority: **low-frequency / optional candidate**. Runtime backend may be replaceable if the interface is preserved.

Argument schemas:

- 170 (100.00%): `{"encoding": "native", "keys": {"amount": "number", "image": "string"}, "type": "object"}`
  - Example: `{"image": "img_1", "amount": 2.0}`

Observation format:

- Types: `{"string": 170}`
- Text/JSON: 100.00% / 0.00%
- Length min/mean/median/max: 331 / 332.0 / 332.0 / 333
- Top-level JSON keys: `{}`
- Text markers: `{}`
- success_like / failure_like / unknown: 170 / 0 / 0

success_like examples:

- `livevqa:68` args=`{"image": "img_1", "amount": 2.0}`; observation: '<image><observation>\nImage sharpening completed successfully. New image ID: img_3. The sharpened image has been uploaded to: http://tj-zw1-303793872-gtzrcx9os6cdhzuhp8egg-1258344706.cos.ap-zhongwei-taiji.myqcloud.com/shawncschen/2026-01-29/shawncschen_fvqa_2026-01-29_fvqa/fvqa_train_7012_trajectory_turn3_sharpen.png\n</observation>'
- `livevqa:141` args=`{"image": "img_1", "amount": 1.5}`; observation: '<image><observation>\nImage sharpening completed successfully. New image ID: img_2. The sharpened image has been uploaded to: http://tj-zw1-303793872-gtzrcx9os6cdhzuhp8egg-1258344706.cos.ap-zhongwei-taiji.myqcloud.com/shawncschen/2026-02-02/shawncschen_fvqa_2026-02-02_fvqa/fvqa_train_7076_trajectory_turn1_sharpen.png\n</observation>'
- `livevqa:236` args=`{"image": "img_2", "amount": 2.0}`; observation: '<image><observation>\nImage sharpening completed successfully. New image ID: img_3. The sharpened image has been uploaded to: http://tj-zw1-303793872-gtzrcx9os6cdhzuhp8egg-1258344706.cos.ap-zhongwei-taiji.myqcloud.com/shawncschen/2026-02-04/shawncschen_fvqa_2026-02-04_fvqa/fvqa_train_8371_trajectory_turn1_sharpen.png\n</observation>'

### `web_search`

Priority: **low-frequency / optional candidate**. Runtime backend may be replaceable if the interface is preserved.

Argument schemas:

- 52 (83.87%): `{"encoding": "native", "keys": {"hl": "string", "q": "string"}, "type": "object"}`
  - Example: `{"q": "Mattel MAT stock open price April 7 2025 nasdaq historical", "hl": "en"}`
- 10 (16.13%): `{"encoding": "native", "keys": {"q": "string"}, "type": "object"}`
  - Example: `{"q": "black and white photo man suit 1960s politician historical figure"}`

Observation format:

- Types: `{"string": 62}`
- Text/JSON: 100.00% / 0.00%
- Length min/mean/median/max: 1794 / 10768.2 / 10714.0 / 14623
- Top-level JSON keys: `{}`
- Text markers: `{"Title:": 62, "URL:": 62, "Content:": 1}`
- success_like / failure_like / unknown: 56 / 6 / 0

success_like examples:

- `fvqa:1804` args=`{"q": "black and white photo man suit 1960s politician historical figure"}`; observation: "<observation>\nTool execution result:\n\n\n============================================================[Passage 1]\nTitle: 1960s Man Suit Politician White House Veranda Steps Working ...\nURL: https://www.ebay.com/itm/127535978786\nSummary:\n<think>\nOkay, let's see. The user wants a summary related to a black and white photo of a man in a suit from the 1960s, specifically a politician or historical figure. The webpage title mentions a 1960s man in a suit, politician, White House veranda steps, working, "
- `fvqa:2039` args=`{"q": "Mattel MAT stock closing price April 7 2025", "hl": "en"}`; observation: '<observation>\nTool execution result:\n\n\n============================================================[Passage 1]\nTitle: Mattel, Inc. (MAT) Stock Historical Prices & Data - Yahoo Finance\nURL: https://finance.yahoo.com/quote/MAT/history/\nSummary:\n<think>\nOkay, let\'s see. The user is asking for the closing price of Mattel\'s stock (MAT) on April 7, 2025. The webpage provided is from Yahoo Finance, which lists historical prices.\n\nLooking at the content, there\'s a line that says "Apr 7, 2025, 15.97, 16.'
- `livevqa:2239` args=`{"q": "India official event ceremony green turban yellow saree stage", "hl": "en"}`; observation: '<observation>\nTool execution result:\n\n\n============================================================[Passage 1]\nTitle: Maharashtrian Wedding Rituals, Dresses And More - WeddingWire.in\nURL: https://www.weddingwire.in/wedding-tips/maharashtrian-wedding--c754\nSummary:\n<think>\nOkay, let\'s tackle this query. The user is asking about an "India official event ceremony green turban yellow saree stage." They probably want to know if there\'s a specific ceremony in Indian events where a green turban and yel'

failure_like examples:

- `fvqa:1693` args=`{"q": "Mattel MAT stock open price April 7 2025 nasdaq historical", "hl": "en"}`; observation: '<observation>\nTool execution result:\n\n\n============================================================[Passage 1]\nTitle: Mattel, Inc. (MAT) Stock Historical Prices & Data - Yahoo Finance\nURL: https://finance.yahoo.com/quote/MAT/history/\nSummary:\n<think>\nOkay, let\'s see. The user is asking for the Mattel (MAT) stock open price on April 7, 2025, from NASDAQ historical data. The provided webpage is from Yahoo Finance, but it shows an error message saying "Oops, something went wrong." So, the content i'
- `livevqa:6292` args=`{"q": "The View ABC daytime talk show broadcast time schedule", "hl": "en"}`; observation: '<observation>\nTool execution result:\n\n\n============================================================[Passage 1]\nTitle: Schedule | The View - TheView.tv\nURL: https://www.theview.tv/schedule\nSummary:\n<think>\nOkay, let\'s see. The user is asking about the broadcast time schedule for The View, an ABC daytime talk show. I need to extract the relevant info from the provided webpage content.\n\nLooking at the content, there\'s a section under "This Week\'s Schedule" with days from Monday to Friday, each with'
- `livevqa:7818` args=`{"q": "Asian woman congressional testimony hearing dark hair microphone", "hl": "en"}`; observation: '<observation>\nTool execution result:\n\n\n============================================================[Passage 1]\nTitle: RAW: Full Hima Kolanagireddy testimony before Michigan House ...\nURL: https://www.youtube.com/watch?v=zG2RkKBHX0M\nSummary:\n<think>\nOkay, let\'s tackle this query. The user wants a concise summary related to "Asian woman congressional testimony hearing dark hair microphone." The webpage is about Hima Kolanagireddy\'s testimony before the Michigan House alleging election fraud.\n\nFirs'

### `perspective_correct`

Priority: **low-frequency / optional candidate**. Runtime backend may be replaceable if the interface is preserved.

Argument schemas:

- 12 (100.00%): `{"encoding": "native", "keys": {"image": "string"}, "type": "object"}`
  - Example: `{"image": "img_1"}`

Observation format:

- Types: `{"string": 12}`
- Text/JSON: 100.00% / 0.00%
- Length min/mean/median/max: 349 / 349.7 / 350.0 / 350
- Top-level JSON keys: `{}`
- Text markers: `{}`
- success_like / failure_like / unknown: 12 / 0 / 0

success_like examples:

- `fvqa:3026` args=`{"image": "img_1"}`; observation: '<image><observation>\nPerspective correction completed successfully. New image ID: img_3. The corrected image has been uploaded to: http://tj-zw1-303793872-gtzrcx9os6cdhzuhp8egg-1258344706.cos.ap-zhongwei-taiji.myqcloud.com/shawncschen/2026-01-06/shawncschen_fvqa_2026-01-06_fvqa/fvqa_train_3848_trajectory_turn1_perspective_correct.png\n</observation>'
- `livevqa:236` args=`{"image": "img_1"}`; observation: '<image><observation>\nPerspective correction completed successfully. New image ID: img_2. The corrected image has been uploaded to: http://tj-zw1-303793872-gtzrcx9os6cdhzuhp8egg-1258344706.cos.ap-zhongwei-taiji.myqcloud.com/shawncschen/2026-02-04/shawncschen_fvqa_2026-02-04_fvqa/fvqa_train_8371_trajectory_turn0_perspective_correct.png\n</observation>'
- `livevqa:9736` args=`{"image": "img_1"}`; observation: '<image><observation>\nPerspective correction completed successfully. New image ID: img_2. The corrected image has been uploaded to: http://tj-zw1-303793872-gtzrcx9os6cdhzuhp8egg-1258344706.cos.ap-zhongwei-taiji.myqcloud.com/shawncschen/2026-01-24/shawncschen_fvqa_2026-01-24_fvqa/fvqa_train_4972_trajectory_turn0_perspective_correct.png\n</observation>'

## 4. Most common tool combinations

- image_search + text_search: 28,404
- crop + image_search + text_search: 1,733
- text_search: 1,676
- image_search: 720
- image_search + layout_parsing + text_search: 678
- layout_parsing + text_search: 617
- crop + image_search + layout_parsing + text_search: 531
- crop + text_search: 342
- crop: 289
- crop + layout_parsing + text_search: 216
- crop + image_search + layout_parsing + super_resolution + text_search: 144
- crop + image_search: 119
- crop + image_search + super_resolution + text_search: 116
- crop + layout_parsing: 97
- image_search + layout_parsing + super_resolution + text_search: 72
- image_search + super_resolution + text_search: 49
- image_search + text_search + web_search: 45
- crop + image_search + sharpen + text_search: 44
- image_search + layout_parsing: 27
- image_search + layout_parsing + sharpen + super_resolution + text_search: 27

## 5. Pairwise co-occurrence

- image_search + text_search: 31,914
- crop + text_search: 3,177
- crop + image_search: 2,750
- layout_parsing + text_search: 2,356
- image_search + layout_parsing: 1,551
- crop + layout_parsing: 1,062
- super_resolution + text_search: 466
- image_search + super_resolution: 451
- layout_parsing + super_resolution: 308
- crop + super_resolution: 300
- image_search + sharpen: 140
- sharpen + text_search: 137
- crop + sharpen: 87
- layout_parsing + sharpen: 81
- image_search + web_search: 58
- text_search + web_search: 57
- sharpen + super_resolution: 55
- image_search + perspective_correct: 12
- perspective_correct + text_search: 11
- crop + web_search: 10

## 6. Adjacent tool transitions

- image_search → text_search: 30,892
- text_search → text_search: 20,047
- layout_parsing → text_search: 1,281
- crop → text_search: 1,143
- crop → image_search: 1,083
- crop → crop: 1,079
- image_search → crop: 1,075
- layout_parsing → image_search: 932
- crop → layout_parsing: 783
- text_search → crop: 418
- layout_parsing → crop: 272
- super_resolution → layout_parsing: 264
- crop → super_resolution: 219
- image_search → layout_parsing: 193
- super_resolution → image_search: 112
- text_search → image_search: 110
- layout_parsing → super_resolution: 98
- super_resolution → text_search: 92
- crop → sharpen: 69
- image_search → super_resolution: 65
- text_search → layout_parsing: 65
- sharpen → layout_parsing: 60
- sharpen → image_search: 47
- text_search → web_search: 41
- sharpen → text_search: 39
- image_search → sharpen: 36
- layout_parsing → sharpen: 28
- web_search → text_search: 27
- image_search → image_search: 24
- super_resolution → sharpen: 21

## 7. Failure and malformed patterns

Malformed trajectories: 0


## 8. Compatibility implications

Priorities below are derived only from trajectory usage frequency, call count, and observed format complexity.
They are not implementation decisions.

- `text_search`: **high priority to preserve**; 3 argument schema(s), 1 observation format(s). A replacement backend should preserve the observed call and observation contract.
- `image_search`: **high priority to preserve**; 1 argument schema(s), 1 observation format(s). A replacement backend should preserve the observed call and observation contract.
- `crop`: **high priority to preserve**; 1 argument schema(s), 1 observation format(s). A replacement backend should preserve the observed call and observation contract.
- `layout_parsing`: **medium priority**; 3 argument schema(s), 1 observation format(s). A replacement backend should preserve the observed call and observation contract.
- `super_resolution`: **medium priority**; 1 argument schema(s), 1 observation format(s). A replacement backend should preserve the observed call and observation contract.
- `sharpen`: **low-frequency / optional candidate**; 1 argument schema(s), 1 observation format(s). A replacement backend should preserve the observed call and observation contract.
- `web_search`: **low-frequency / optional candidate**; 2 argument schema(s), 1 observation format(s). A replacement backend should preserve the observed call and observation contract.
- `perspective_correct`: **low-frequency / optional candidate**; 1 argument schema(s), 1 observation format(s). A replacement backend should preserve the observed call and observation contract.

### Focused contract conclusions

#### `image_search`

- Arguments (33,161 calls): `{"encoding": "native", "keys": {"url": "string"}, "type": "object"}`
- Observation formats: `{"text": 33161}`
- Detected text markers: `{"Search results": 20095, "Title:": 3951, "URL:": 857}`
- Multiple argument schemas observed: `false`
- Backend compatibility: Preserve the observed `url` string argument (whose values may be dataset image references such as img_1) and a plain-text search-results observation. Do not silently replace the observation with a JSON-only contract.

#### `layout_parsing`

- Arguments (2,770 calls): `{"encoding": "native", "keys": {"image": "string"}, "type": "object"}`
- Arguments (34 calls): `{"encoding": "native", "keys": {"image": "string", "use_chart_recognition": "boolean"}, "type": "object"}`
- Arguments (4 calls): `{"encoding": "native", "keys": {"image": "string", "use_doc_orientation_classify": "boolean"}, "type": "object"}`
- Observation formats: `{"text": 2808}`
- Detected text markers: `{"Content:": 942, "URL:": 3, "Title:": 1}`
- Multiple argument schemas observed: `true`
- Backend compatibility: Preserve the required `image` string argument, accept calls with no optional flags, and support the observed boolean `use_chart_recognition` and `use_doc_orientation_classify` variants. Preserve a plain-text parsing result; `Content:` is the most common detected structural marker.

The current reproduction repository implements no runtime tool backends yet. Therefore every observed tool is
reported as observed-but-not-implemented; this audit does not add any backend.

## 9. Reproducibility

- Implementation SHA-256: `66abba920cb4bac57fa71a7bb29d0ae69f516386ae75122d0cc87fb42987064c`
- Git commit at generation: `0e334d7c4420f20d30d99da1c240676ee6348fe8`
- Generated at: `2026-09-21T14:18:04.734171+00:00`

- `fvqa`: 4,413 trajectories, SHA-256 `71cea17526a232c7cb91d4e68d4da6ef1249fda669b0adf42df11c7f948f71dd`
- `livevqa`: 13,326 trajectories, SHA-256 `410eb192e4db7af43d71c3cb7379b8bd372a4c1a87421a1b40be5dc086152657`
- `palace`: 2,973 trajectories, SHA-256 `3f2ba5574e2e0f3e1e815023ae09b2a0574bef7696613b2bbab7ba8ed9e1298f`
- `webqa`: 3,804 trajectories, SHA-256 `b922293d16296106ffb8f32c11203ccf669d8d4c007c50c2dbf08fdfc5dbdd6c`
- `wiki_art`: 5,093 trajectories, SHA-256 `6c50fb9b970918ab5d1470c6f75d8c583e7dd284cbaeae9b8a16349a7ea31d1b`
- `wiki_en`: 3,503 trajectories, SHA-256 `a22a44c6a04d79d6dfd0064c89d8a792045278eed70a8e27c14b7c5e2f4850e3`
- `wiki_zh`: 3,480 trajectories, SHA-256 `eacdd6aaaa63f6ea9d513a3fa310178489a947d56def326020df2e0a4e9a4b65`
