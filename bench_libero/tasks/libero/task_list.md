# LIBERO Object-Flow 任务清单（130 tasks）

每个任务从 LIBERO 一条示范中抽取**单个被操作物体**的世界坐标轨迹（object-flow），重锚到 DexVerse 机器人基座系后用于 flow 可视化。下面按 5 个套件分组。

- 总数：**130** 个任务
- 涉及物体（去尾号，共 22 种）：akita_black_bowl, alphabet_soup, bbq_sauce, black_book, butter, chefmate_8_frypan, chocolate_pudding, cream_cheese, ketchup, milk, moka_pot, new_salad_dressing, orange_juice, plate, porcelain_mug, red_coffee_mug, salad_dressing, tomato_sauce, white_bowl, white_yellow_mug, wine_bottle, yellow_book
- 数据源：yifengzhu-hf/LIBERO-datasets (hf-mirror)
- 帧数 = 该示范的时间步数（object-flow 轨迹长度 K）


## libero_spatial （10 个）

| # | 场景 | 任务描述 | 物体 | 帧数 |
|---|------|----------|------|------|
| 1 |  | pick up the black bowl between the plate and the ramekin and place it on the plate | `akita_black_bowl_1` | 98 |
| 2 |  | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | `akita_black_bowl_1` | 153 |
| 3 |  | pick up the black bowl from table center and place it on the plate | `akita_black_bowl_1` | 103 |
| 4 |  | pick up the black bowl next to the plate and place it on the plate | `akita_black_bowl_1` | 115 |
| 5 |  | pick up the black bowl next to the cookie box and place it on the plate | `akita_black_bowl_1` | 123 |
| 6 |  | pick up the black bowl next to the ramekin and place it on the plate | `akita_black_bowl_1` | 152 |
| 7 |  | pick up the black bowl on the ramekin and place it on the plate | `akita_black_bowl_1` | 137 |
| 8 |  | pick up the black bowl on the stove and place it on the plate | `akita_black_bowl_1` | 155 |
| 9 |  | pick up the black bowl on the cookie box and place it on the plate | `akita_black_bowl_1` | 99 |
| 10 |  | pick up the black bowl on the wooden cabinet and place it on the plate | `akita_black_bowl_1` | 136 |

## libero_object （10 个）

| # | 场景 | 任务描述 | 物体 | 帧数 |
|---|------|----------|------|------|
| 11 |  | pick up the chocolate pudding and place it in the basket | `chocolate_pudding_1` | 168 |
| 12 |  | pick up the bbq sauce and place it in the basket | `bbq_sauce_1` | 134 |
| 13 |  | pick up the butter and place it in the basket | `butter_1` | 169 |
| 14 |  | pick up the alphabet soup and place it in the basket | `alphabet_soup_1` | 148 |
| 15 |  | pick up the cream cheese and place it in the basket | `cream_cheese_1` | 139 |
| 16 |  | pick up the salad dressing and place it in the basket | `salad_dressing_1` | 137 |
| 17 |  | pick up the orange juice and place it in the basket | `orange_juice_1` | 126 |
| 18 |  | pick up the milk and place it in the basket | `milk_1` | 150 |
| 19 |  | pick up the ketchup and place it in the basket | `ketchup_1` | 248 |
| 20 |  | pick up the tomato sauce and place it in the basket | `tomato_sauce_1` | 132 |

## libero_goal （10 个）

| # | 场景 | 任务描述 | 物体 | 帧数 |
|---|------|----------|------|------|
| 21 |  | open the top drawer and put the bowl inside | `akita_black_bowl_1` | 170 |
| 22 |  | open the middle drawer of the cabinet | `akita_black_bowl_1` | 138 |
| 23 |  | put the bowl on the stove | `akita_black_bowl_1` | 94 |
| 24 |  | put the bowl on the plate | `akita_black_bowl_1` | 90 |
| 25 |  | put the cream cheese in the bowl | `cream_cheese_1` | 92 |
| 26 |  | push the plate to the front of the stove | `plate_1` | 155 |
| 27 |  | put the bowl on top of the cabinet | `akita_black_bowl_1` | 91 |
| 28 |  | turn on the stove | `akita_black_bowl_1` | 80 |
| 29 |  | put the wine bottle on the rack | `wine_bottle_1` | 347 |
| 30 |  | put the wine bottle on top of the cabinet | `wine_bottle_1` | 93 |

## libero_10 （10 个）

| # | 场景 | 任务描述 | 物体 | 帧数 |
|---|------|----------|------|------|
| 31 | KITCHEN_SCENE8 | put both moka pots on the stove | `moka_pot_1` | 457 |
| 32 | LIVING_ROOM_SCENE2 | put both the cream cheese box and the butter in the basket | `cream_cheese_1` | 258 |
| 33 | LIVING_ROOM_SCENE1 | put both the alphabet soup and the cream cheese box in the basket | `alphabet_soup_1` | 242 |
| 34 | KITCHEN_SCENE4 | put the black bowl in the bottom drawer of the cabinet and close it | `akita_black_bowl_1` | 261 |
| 35 | KITCHEN_SCENE6 | put the yellow and white mug in the microwave and close it | `white_yellow_mug_1` | 329 |
| 36 | LIVING_ROOM_SCENE5 | put the white mug on the left plate and put the yellow and white mug on the right plate | `white_yellow_mug_1` | 275 |
| 37 | LIVING_ROOM_SCENE2 | put both the alphabet soup and the tomato sauce in the basket | `alphabet_soup_1` | 388 |
| 38 | KITCHEN_SCENE3 | turn on the stove and put the moka pot on it | `moka_pot_1` | 272 |
| 39 | STUDY_SCENE1 | pick up the book and place it in the back compartment of the caddy | `black_book_1` | 234 |
| 40 | LIVING_ROOM_SCENE6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | `porcelain_mug_1` | 203 |

## libero_90 （90 个）

| # | 场景 | 任务描述 | 物体 | 帧数 |
|---|------|----------|------|------|
| 41 | KITCHEN_SCENE10 | close the top drawer of the cabinet and put the black bowl on top of it | `akita_black_bowl_1` | 197 |
| 42 | KITCHEN_SCENE10 | put the black bowl in the top drawer of the cabinet | `akita_black_bowl_1` | 116 |
| 43 | KITCHEN_SCENE10 | put the butter at the front in the top drawer of the cabinet and close it | `butter_1` | 173 |
| 44 | KITCHEN_SCENE10 | put the butter at the back in the top drawer of the cabinet and close it | `butter_2` | 183 |
| 45 | KITCHEN_SCENE10 | close the top drawer of the cabinet | `akita_black_bowl_1` | 71 |
| 46 | KITCHEN_SCENE10 | put the chocolate pudding in the top drawer of the cabinet and close it | `chocolate_pudding_1` | 168 |
| 47 | KITCHEN_SCENE1 | open the top drawer of the cabinet | `akita_black_bowl_1` | 177 |
| 48 | KITCHEN_SCENE1 | open the bottom drawer of the cabinet | `akita_black_bowl_1` | 124 |
| 49 | KITCHEN_SCENE2 | open the top drawer of the cabinet | `akita_black_bowl_1` | 64 |
| 50 | KITCHEN_SCENE1 | open the top drawer of the cabinet and put the bowl in it | `akita_black_bowl_1` | 189 |
| 51 | KITCHEN_SCENE2 | put the black bowl at the front on the plate | `akita_black_bowl_1` | 134 |
| 52 | KITCHEN_SCENE1 | put the black bowl on top of the cabinet | `akita_black_bowl_1` | 161 |
| 53 | KITCHEN_SCENE2 | put the black bowl at the back on the plate | `akita_black_bowl_3` | 108 |
| 54 | KITCHEN_SCENE2 | put the middle black bowl on the plate | `akita_black_bowl_2` | 108 |
| 55 | KITCHEN_SCENE2 | put the middle black bowl on top of the cabinet | `akita_black_bowl_2` | 137 |
| 56 | KITCHEN_SCENE1 | put the black bowl on the plate | `akita_black_bowl_1` | 137 |
| 57 | KITCHEN_SCENE2 | stack the black bowl at the front on the black bowl in the middle | `akita_black_bowl_1` | 119 |
| 58 | KITCHEN_SCENE2 | stack the middle black bowl on the back black bowl | `akita_black_bowl_2` | 137 |
| 59 | KITCHEN_SCENE3 | put the frying pan on the stove | `chefmate_8_frypan_1` | 184 |
| 60 | KITCHEN_SCENE3 | turn on the stove | `chefmate_8_frypan_1` | 90 |
| 61 | KITCHEN_SCENE3 | put the moka pot on the stove | `moka_pot_1` | 135 |
| 62 | KITCHEN_SCENE4 | close the bottom drawer of the cabinet and open the top drawer | `akita_black_bowl_1` | 217 |
| 63 | KITCHEN_SCENE4 | close the bottom drawer of the cabinet | `akita_black_bowl_1` | 137 |
| 64 | KITCHEN_SCENE4 | put the black bowl in the bottom drawer of the cabinet | `akita_black_bowl_1` | 166 |
| 65 | KITCHEN_SCENE4 | put the black bowl on top of the cabinet | `akita_black_bowl_1` | 147 |
| 66 | KITCHEN_SCENE3 | turn on the stove and put the frying pan on it | `chefmate_8_frypan_1` | 287 |
| 67 | KITCHEN_SCENE5 | close the top drawer of the cabinet | `akita_black_bowl_1` | 65 |
| 68 | KITCHEN_SCENE4 | put the wine bottle on the wine rack | `wine_bottle_1` | 309 |
| 69 | KITCHEN_SCENE5 | put the black bowl in the top drawer of the cabinet | `akita_black_bowl_1` | 122 |
| 70 | KITCHEN_SCENE4 | put the wine bottle in the bottom drawer of the cabinet | `wine_bottle_1` | 169 |
| 71 | KITCHEN_SCENE5 | put the black bowl on the plate | `akita_black_bowl_1` | 131 |
| 72 | KITCHEN_SCENE5 | put the black bowl on top of the cabinet | `akita_black_bowl_1` | 147 |
| 73 | KITCHEN_SCENE5 | put the ketchup in the top drawer of the cabinet | `ketchup_1` | 230 |
| 74 | KITCHEN_SCENE6 | put the yellow and white mug to the front of the white mug | `white_yellow_mug_1` | 144 |
| 75 | KITCHEN_SCENE6 | close the microwave | `white_yellow_mug_1` | 236 |
| 76 | KITCHEN_SCENE7 | put the white bowl to the right of the plate | `white_bowl_1` | 120 |
| 77 | KITCHEN_SCENE8 | turn off the stove | `moka_pot_1` | 152 |
| 78 | KITCHEN_SCENE7 | put the white bowl on the plate | `white_bowl_1` | 223 |
| 79 | KITCHEN_SCENE8 | put the right moka pot on the stove | `moka_pot_1` | 192 |
| 80 | KITCHEN_SCENE9 | put the frying pan on top of the cabinet | `chefmate_8_frypan_1` | 227 |
| 81 | KITCHEN_SCENE9 | put the frying pan on the cabinet shelf | `chefmate_8_frypan_1` | 177 |
| 82 | KITCHEN_SCENE7 | open the microwave | `white_bowl_1` | 131 |
| 83 | KITCHEN_SCENE9 | put the frying pan under the cabinet shelf | `chefmate_8_frypan_1` | 172 |
| 84 | KITCHEN_SCENE9 | put the white bowl on top of the cabinet | `white_bowl_1` | 144 |
| 85 | KITCHEN_SCENE9 | turn on the stove | `white_bowl_1` | 115 |
| 86 | LIVING_ROOM_SCENE2 | pick up the alphabet soup and put it in the basket | `alphabet_soup_1` | 164 |
| 87 | LIVING_ROOM_SCENE1 | pick up the ketchup and put it in the basket | `ketchup_1` | 173 |
| 88 | LIVING_ROOM_SCENE1 | pick up the tomato sauce and put it in the basket | `tomato_sauce_1` | 162 |
| 89 | LIVING_ROOM_SCENE1 | pick up the cream cheese box and put it in the basket | `cream_cheese_1` | 161 |
| 90 | LIVING_ROOM_SCENE2 | pick up the milk and put it in the basket | `milk_1` | 104 |
| 91 | KITCHEN_SCENE9 | turn on the stove and put the frying pan on it | `chefmate_8_frypan_1` | 255 |
| 92 | LIVING_ROOM_SCENE1 | pick up the alphabet soup and put it in the basket | `alphabet_soup_1` | 154 |
| 93 | LIVING_ROOM_SCENE2 | pick up the butter and put it in the basket | `butter_1` | 146 |
| 94 | LIVING_ROOM_SCENE2 | pick up the tomato sauce and put it in the basket | `tomato_sauce_1` | 104 |
| 95 | LIVING_ROOM_SCENE3 | pick up the butter and put it in the tray | `butter_1` | 103 |
| 96 | LIVING_ROOM_SCENE3 | pick up the alphabet soup and put it in the tray | `alphabet_soup_1` | 111 |
| 97 | LIVING_ROOM_SCENE3 | pick up the tomato sauce and put it in the tray | `tomato_sauce_1` | 128 |
| 98 | LIVING_ROOM_SCENE3 | pick up the cream cheese and put it in the tray | `cream_cheese_1` | 158 |
| 99 | LIVING_ROOM_SCENE3 | pick up the ketchup and put it in the tray | `ketchup_1` | 143 |
| 100 | LIVING_ROOM_SCENE2 | pick up the orange juice and put it in the basket | `orange_juice_1` | 148 |
| 101 | LIVING_ROOM_SCENE4 | stack the left bowl on the right bowl and place them in the tray | `akita_black_bowl_1` | 206 |
| 102 | LIVING_ROOM_SCENE4 | stack the right bowl on the left bowl and place them in the tray | `akita_black_bowl_2` | 231 |
| 103 | LIVING_ROOM_SCENE4 | pick up the chocolate pudding and put it in the tray | `chocolate_pudding_1` | 151 |
| 104 | LIVING_ROOM_SCENE5 | put the red mug on the left plate | `red_coffee_mug_1` | 127 |
| 105 | LIVING_ROOM_SCENE5 | put the white mug on the left plate | `porcelain_mug_1` | 88 |
| 106 | LIVING_ROOM_SCENE5 | put the red mug on the right plate | `red_coffee_mug_1` | 180 |
| 107 | LIVING_ROOM_SCENE4 | pick up the salad dressing and put it in the tray | `new_salad_dressing_1` | 114 |
| 108 | LIVING_ROOM_SCENE5 | put the yellow and white mug on the right plate | `white_yellow_mug_1` | 121 |
| 109 | LIVING_ROOM_SCENE4 | pick up the black bowl on the left and put it in the tray | `akita_black_bowl_1` | 110 |
| 110 | LIVING_ROOM_SCENE6 | put the chocolate pudding to the left of the plate | `chocolate_pudding_1` | 110 |
| 111 | LIVING_ROOM_SCENE6 | put the chocolate pudding to the right of the plate | `chocolate_pudding_1` | 77 |
| 112 | LIVING_ROOM_SCENE6 | put the red mug on the plate | `red_coffee_mug_1` | 116 |
| 113 | LIVING_ROOM_SCENE6 | put the white mug on the plate | `porcelain_mug_1` | 147 |
| 114 | STUDY_SCENE1 | pick up the book and place it in the left compartment of the caddy | `black_book_1` | 148 |
| 115 | STUDY_SCENE1 | pick up the yellow and white mug and place it to the right of the caddy | `white_yellow_mug_1` | 129 |
| 116 | STUDY_SCENE2 | pick up the book and place it in the back compartment of the caddy | `black_book_1` | 197 |
| 117 | STUDY_SCENE1 | pick up the book and place it in the right compartment of the caddy | `black_book_1` | 142 |
| 118 | STUDY_SCENE2 | pick up the book and place it in the right compartment of the caddy | `black_book_1` | 161 |
| 119 | STUDY_SCENE1 | pick up the book and place it in the front compartment of the caddy | `black_book_1` | 165 |
| 120 | STUDY_SCENE2 | pick up the book and place it in the left compartment of the caddy | `black_book_1` | 108 |
| 121 | STUDY_SCENE2 | pick up the book and place it in the front compartment of the caddy | `black_book_1` | 133 |
| 122 | STUDY_SCENE3 | pick up the book and place it in the right compartment of the caddy | `black_book_1` | 121 |
| 123 | STUDY_SCENE3 | pick up the book and place it in the left compartment of the caddy | `black_book_1` | 159 |
| 124 | STUDY_SCENE3 | pick up the book and place it in the front compartment of the caddy | `black_book_1` | 139 |
| 125 | STUDY_SCENE3 | pick up the white mug and place it to the right of the caddy | `porcelain_mug_1` | 112 |
| 126 | STUDY_SCENE3 | pick up the red mug and place it to the right of the caddy | `red_coffee_mug_1` | 119 |
| 127 | STUDY_SCENE4 | pick up the book on the right and place it on the cabinet shelf | `yellow_book_1` | 92 |
| 128 | STUDY_SCENE4 | pick up the book in the middle and place it on the cabinet shelf | `black_book_1` | 167 |
| 129 | STUDY_SCENE4 | pick up the book on the left and place it on top of the shelf | `yellow_book_2` | 174 |
| 130 | STUDY_SCENE4 | pick up the book on the right and place it under the cabinet shelf | `yellow_book_1` | 123 |
