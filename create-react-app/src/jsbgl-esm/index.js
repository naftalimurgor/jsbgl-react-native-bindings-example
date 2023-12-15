import BN from 'bn.js';
import { Buffer as Buffer$1 } from 'buffer/';

function mnemonicWordlist(window) {
  window.BIP39_WORDLIST = ["abandon",
    "ability",
    "able",
    "about",
    "above",
    "absent",
    "absorb",
    "abstract",
    "absurd",
    "abuse",
    "access",
    "accident",
    "account",
    "accuse",
    "achieve",
    "acid",
    "acoustic",
    "acquire",
    "across",
    "act",
    "action",
    "actor",
    "actress",
    "actual",
    "adapt",
    "add",
    "addict",
    "address",
    "adjust",
    "admit",
    "adult",
    "advance",
    "advice",
    "aerobic",
    "affair",
    "afford",
    "afraid",
    "again",
    "age",
    "agent",
    "agree",
    "ahead",
    "aim",
    "air",
    "airport",
    "aisle",
    "alarm",
    "album",
    "alcohol",
    "alert",
    "alien",
    "all",
    "alley",
    "allow",
    "almost",
    "alone",
    "alpha",
    "already",
    "also",
    "alter",
    "always",
    "amateur",
    "amazing",
    "among",
    "amount",
    "amused",
    "analyst",
    "anchor",
    "ancient",
    "anger",
    "angle",
    "angry",
    "animal",
    "ankle",
    "announce",
    "annual",
    "another",
    "answer",
    "antenna",
    "antique",
    "anxiety",
    "any",
    "apart",
    "apology",
    "appear",
    "apple",
    "approve",
    "april",
    "arch",
    "arctic",
    "area",
    "arena",
    "argue",
    "arm",
    "armed",
    "armor",
    "army",
    "around",
    "arrange",
    "arrest",
    "arrive",
    "arrow",
    "art",
    "artefact",
    "artist",
    "artwork",
    "ask",
    "aspect",
    "assault",
    "asset",
    "assist",
    "assume",
    "asthma",
    "athlete",
    "atom",
    "attack",
    "attend",
    "attitude",
    "attract",
    "auction",
    "audit",
    "august",
    "aunt",
    "author",
    "auto",
    "autumn",
    "average",
    "avocado",
    "avoid",
    "awake",
    "aware",
    "away",
    "awesome",
    "awful",
    "awkward",
    "axis",
    "baby",
    "bachelor",
    "bacon",
    "badge",
    "bag",
    "balance",
    "balcony",
    "ball",
    "bamboo",
    "banana",
    "banner",
    "bar",
    "barely",
    "bargain",
    "barrel",
    "base",
    "basic",
    "basket",
    "battle",
    "beach",
    "bean",
    "beauty",
    "because",
    "become",
    "beef",
    "before",
    "begin",
    "behave",
    "behind",
    "believe",
    "below",
    "belt",
    "bench",
    "benefit",
    "best",
    "betray",
    "better",
    "between",
    "beyond",
    "bicycle",
    "bid",
    "bike",
    "bind",
    "biology",
    "bird",
    "birth",
    "bitter",
    "black",
    "blade",
    "blame",
    "blanket",
    "blast",
    "bleak",
    "bless",
    "blind",
    "blood",
    "blossom",
    "blouse",
    "blue",
    "blur",
    "blush",
    "board",
    "boat",
    "body",
    "boil",
    "bomb",
    "bone",
    "bonus",
    "book",
    "boost",
    "border",
    "boring",
    "borrow",
    "boss",
    "bottom",
    "bounce",
    "box",
    "boy",
    "bracket",
    "brain",
    "brand",
    "brass",
    "brave",
    "bread",
    "breeze",
    "brick",
    "bridge",
    "brief",
    "bright",
    "bring",
    "brisk",
    "broccoli",
    "broken",
    "bronze",
    "broom",
    "brother",
    "brown",
    "brush",
    "bubble",
    "buddy",
    "budget",
    "buffalo",
    "build",
    "bulb",
    "bulk",
    "bullet",
    "bundle",
    "bunker",
    "burden",
    "burger",
    "burst",
    "bus",
    "business",
    "busy",
    "butter",
    "buyer",
    "buzz",
    "cabbage",
    "cabin",
    "cable",
    "cactus",
    "cage",
    "cake",
    "call",
    "calm",
    "camera",
    "camp",
    "can",
    "canal",
    "cancel",
    "candy",
    "cannon",
    "canoe",
    "canvas",
    "canyon",
    "capable",
    "capital",
    "captain",
    "car",
    "carbon",
    "card",
    "cargo",
    "carpet",
    "carry",
    "cart",
    "case",
    "cash",
    "casino",
    "castle",
    "casual",
    "cat",
    "catalog",
    "catch",
    "category",
    "cattle",
    "caught",
    "cause",
    "caution",
    "cave",
    "ceiling",
    "celery",
    "cement",
    "census",
    "century",
    "cereal",
    "certain",
    "chair",
    "chalk",
    "champion",
    "change",
    "chaos",
    "chapter",
    "charge",
    "chase",
    "chat",
    "cheap",
    "check",
    "cheese",
    "chef",
    "cherry",
    "chest",
    "chicken",
    "chief",
    "child",
    "chimney",
    "choice",
    "choose",
    "chronic",
    "chuckle",
    "chunk",
    "churn",
    "cigar",
    "cinnamon",
    "circle",
    "citizen",
    "city",
    "civil",
    "claim",
    "clap",
    "clarify",
    "claw",
    "clay",
    "clean",
    "clerk",
    "clever",
    "click",
    "client",
    "cliff",
    "climb",
    "clinic",
    "clip",
    "clock",
    "clog",
    "close",
    "cloth",
    "cloud",
    "clown",
    "club",
    "clump",
    "cluster",
    "clutch",
    "coach",
    "coast",
    "coconut",
    "code",
    "coffee",
    "coil",
    "coin",
    "collect",
    "color",
    "column",
    "combine",
    "come",
    "comfort",
    "comic",
    "common",
    "company",
    "concert",
    "conduct",
    "confirm",
    "congress",
    "connect",
    "consider",
    "control",
    "convince",
    "cook",
    "cool",
    "copper",
    "copy",
    "coral",
    "core",
    "corn",
    "correct",
    "cost",
    "cotton",
    "couch",
    "country",
    "couple",
    "course",
    "cousin",
    "cover",
    "coyote",
    "crack",
    "cradle",
    "craft",
    "cram",
    "crane",
    "crash",
    "crater",
    "crawl",
    "crazy",
    "cream",
    "credit",
    "creek",
    "crew",
    "cricket",
    "crime",
    "crisp",
    "critic",
    "crop",
    "cross",
    "crouch",
    "crowd",
    "crucial",
    "cruel",
    "cruise",
    "crumble",
    "crunch",
    "crush",
    "cry",
    "crystal",
    "cube",
    "culture",
    "cup",
    "cupboard",
    "curious",
    "current",
    "curtain",
    "curve",
    "cushion",
    "custom",
    "cute",
    "cycle",
    "dad",
    "damage",
    "damp",
    "dance",
    "danger",
    "daring",
    "dash",
    "daughter",
    "dawn",
    "day",
    "deal",
    "debate",
    "debris",
    "decade",
    "december",
    "decide",
    "decline",
    "decorate",
    "decrease",
    "deer",
    "defense",
    "define",
    "defy",
    "degree",
    "delay",
    "deliver",
    "demand",
    "demise",
    "denial",
    "dentist",
    "deny",
    "depart",
    "depend",
    "deposit",
    "depth",
    "deputy",
    "derive",
    "describe",
    "desert",
    "design",
    "desk",
    "despair",
    "destroy",
    "detail",
    "detect",
    "develop",
    "device",
    "devote",
    "diagram",
    "dial",
    "diamond",
    "diary",
    "dice",
    "diesel",
    "diet",
    "differ",
    "digital",
    "dignity",
    "dilemma",
    "dinner",
    "dinosaur",
    "direct",
    "dirt",
    "disagree",
    "discover",
    "disease",
    "dish",
    "dismiss",
    "disorder",
    "display",
    "distance",
    "divert",
    "divide",
    "divorce",
    "dizzy",
    "doctor",
    "document",
    "dog",
    "doll",
    "dolphin",
    "domain",
    "donate",
    "donkey",
    "donor",
    "door",
    "dose",
    "double",
    "dove",
    "draft",
    "dragon",
    "drama",
    "drastic",
    "draw",
    "dream",
    "dress",
    "drift",
    "drill",
    "drink",
    "drip",
    "drive",
    "drop",
    "drum",
    "dry",
    "duck",
    "dumb",
    "dune",
    "during",
    "dust",
    "dutch",
    "duty",
    "dwarf",
    "dynamic",
    "eager",
    "eagle",
    "early",
    "earn",
    "earth",
    "easily",
    "east",
    "easy",
    "echo",
    "ecology",
    "economy",
    "edge",
    "edit",
    "educate",
    "effort",
    "egg",
    "eight",
    "either",
    "elbow",
    "elder",
    "electric",
    "elegant",
    "element",
    "elephant",
    "elevator",
    "elite",
    "else",
    "embark",
    "embody",
    "embrace",
    "emerge",
    "emotion",
    "employ",
    "empower",
    "empty",
    "enable",
    "enact",
    "end",
    "endless",
    "endorse",
    "enemy",
    "energy",
    "enforce",
    "engage",
    "engine",
    "enhance",
    "enjoy",
    "enlist",
    "enough",
    "enrich",
    "enroll",
    "ensure",
    "enter",
    "entire",
    "entry",
    "envelope",
    "episode",
    "equal",
    "equip",
    "era",
    "erase",
    "erode",
    "erosion",
    "error",
    "erupt",
    "escape",
    "essay",
    "essence",
    "estate",
    "eternal",
    "ethics",
    "evidence",
    "evil",
    "evoke",
    "evolve",
    "exact",
    "example",
    "excess",
    "exchange",
    "excite",
    "exclude",
    "excuse",
    "execute",
    "exercise",
    "exhaust",
    "exhibit",
    "exile",
    "exist",
    "exit",
    "exotic",
    "expand",
    "expect",
    "expire",
    "explain",
    "expose",
    "express",
    "extend",
    "extra",
    "eye",
    "eyebrow",
    "fabric",
    "face",
    "faculty",
    "fade",
    "faint",
    "faith",
    "fall",
    "false",
    "fame",
    "family",
    "famous",
    "fan",
    "fancy",
    "fantasy",
    "farm",
    "fashion",
    "fat",
    "fatal",
    "father",
    "fatigue",
    "fault",
    "favorite",
    "feature",
    "february",
    "federal",
    "fee",
    "feed",
    "feel",
    "female",
    "fence",
    "festival",
    "fetch",
    "fever",
    "few",
    "fiber",
    "fiction",
    "field",
    "figure",
    "file",
    "film",
    "filter",
    "final",
    "find",
    "fine",
    "finger",
    "finish",
    "fire",
    "firm",
    "first",
    "fiscal",
    "fish",
    "fit",
    "fitness",
    "fix",
    "flag",
    "flame",
    "flash",
    "flat",
    "flavor",
    "flee",
    "flight",
    "flip",
    "float",
    "flock",
    "floor",
    "flower",
    "fluid",
    "flush",
    "fly",
    "foam",
    "focus",
    "fog",
    "foil",
    "fold",
    "follow",
    "food",
    "foot",
    "force",
    "forest",
    "forget",
    "fork",
    "fortune",
    "forum",
    "forward",
    "fossil",
    "foster",
    "found",
    "fox",
    "fragile",
    "frame",
    "frequent",
    "fresh",
    "friend",
    "fringe",
    "frog",
    "front",
    "frost",
    "frown",
    "frozen",
    "fruit",
    "fuel",
    "fun",
    "funny",
    "furnace",
    "fury",
    "future",
    "gadget",
    "gain",
    "galaxy",
    "gallery",
    "game",
    "gap",
    "garage",
    "garbage",
    "garden",
    "garlic",
    "garment",
    "gas",
    "gasp",
    "gate",
    "gather",
    "gauge",
    "gaze",
    "general",
    "genius",
    "genre",
    "gentle",
    "genuine",
    "gesture",
    "ghost",
    "giant",
    "gift",
    "giggle",
    "ginger",
    "giraffe",
    "girl",
    "give",
    "glad",
    "glance",
    "glare",
    "glass",
    "glide",
    "glimpse",
    "globe",
    "gloom",
    "glory",
    "glove",
    "glow",
    "glue",
    "goat",
    "goddess",
    "gold",
    "good",
    "goose",
    "gorilla",
    "gospel",
    "gossip",
    "govern",
    "gown",
    "grab",
    "grace",
    "grain",
    "grant",
    "grape",
    "grass",
    "gravity",
    "great",
    "green",
    "grid",
    "grief",
    "grit",
    "grocery",
    "group",
    "grow",
    "grunt",
    "guard",
    "guess",
    "guide",
    "guilt",
    "guitar",
    "gun",
    "gym",
    "habit",
    "hair",
    "half",
    "hammer",
    "hamster",
    "hand",
    "happy",
    "harbor",
    "hard",
    "harsh",
    "harvest",
    "hat",
    "have",
    "hawk",
    "hazard",
    "head",
    "health",
    "heart",
    "heavy",
    "hedgehog",
    "height",
    "hello",
    "helmet",
    "help",
    "hen",
    "hero",
    "hidden",
    "high",
    "hill",
    "hint",
    "hip",
    "hire",
    "history",
    "hobby",
    "hockey",
    "hold",
    "hole",
    "holiday",
    "hollow",
    "home",
    "honey",
    "hood",
    "hope",
    "horn",
    "horror",
    "horse",
    "hospital",
    "host",
    "hotel",
    "hour",
    "hover",
    "hub",
    "huge",
    "human",
    "humble",
    "humor",
    "hundred",
    "hungry",
    "hunt",
    "hurdle",
    "hurry",
    "hurt",
    "husband",
    "hybrid",
    "ice",
    "icon",
    "idea",
    "identify",
    "idle",
    "ignore",
    "ill",
    "illegal",
    "illness",
    "image",
    "imitate",
    "immense",
    "immune",
    "impact",
    "impose",
    "improve",
    "impulse",
    "inch",
    "include",
    "income",
    "increase",
    "index",
    "indicate",
    "indoor",
    "industry",
    "infant",
    "inflict",
    "inform",
    "inhale",
    "inherit",
    "initial",
    "inject",
    "injury",
    "inmate",
    "inner",
    "innocent",
    "input",
    "inquiry",
    "insane",
    "insect",
    "inside",
    "inspire",
    "install",
    "intact",
    "interest",
    "into",
    "invest",
    "invite",
    "involve",
    "iron",
    "island",
    "isolate",
    "issue",
    "item",
    "ivory",
    "jacket",
    "jaguar",
    "jar",
    "jazz",
    "jealous",
    "jeans",
    "jelly",
    "jewel",
    "job",
    "join",
    "joke",
    "journey",
    "joy",
    "judge",
    "juice",
    "jump",
    "jungle",
    "junior",
    "junk",
    "just",
    "kangaroo",
    "keen",
    "keep",
    "ketchup",
    "key",
    "kick",
    "kid",
    "kidney",
    "kind",
    "kingdom",
    "kiss",
    "kit",
    "kitchen",
    "kite",
    "kitten",
    "kiwi",
    "knee",
    "knife",
    "knock",
    "know",
    "lab",
    "label",
    "labor",
    "ladder",
    "lady",
    "lake",
    "lamp",
    "language",
    "laptop",
    "large",
    "later",
    "latin",
    "laugh",
    "laundry",
    "lava",
    "law",
    "lawn",
    "lawsuit",
    "layer",
    "lazy",
    "leader",
    "leaf",
    "learn",
    "leave",
    "lecture",
    "left",
    "leg",
    "legal",
    "legend",
    "leisure",
    "lemon",
    "lend",
    "length",
    "lens",
    "leopard",
    "lesson",
    "letter",
    "level",
    "liar",
    "liberty",
    "library",
    "license",
    "life",
    "lift",
    "light",
    "like",
    "limb",
    "limit",
    "link",
    "lion",
    "liquid",
    "list",
    "little",
    "live",
    "lizard",
    "load",
    "loan",
    "lobster",
    "local",
    "lock",
    "logic",
    "lonely",
    "long",
    "loop",
    "lottery",
    "loud",
    "lounge",
    "love",
    "loyal",
    "lucky",
    "luggage",
    "lumber",
    "lunar",
    "lunch",
    "luxury",
    "lyrics",
    "machine",
    "mad",
    "magic",
    "magnet",
    "maid",
    "mail",
    "main",
    "major",
    "make",
    "mammal",
    "man",
    "manage",
    "mandate",
    "mango",
    "mansion",
    "manual",
    "maple",
    "marble",
    "march",
    "margin",
    "marine",
    "market",
    "marriage",
    "mask",
    "mass",
    "master",
    "match",
    "material",
    "math",
    "matrix",
    "matter",
    "maximum",
    "maze",
    "meadow",
    "mean",
    "measure",
    "meat",
    "mechanic",
    "medal",
    "media",
    "melody",
    "melt",
    "member",
    "memory",
    "mention",
    "menu",
    "mercy",
    "merge",
    "merit",
    "merry",
    "mesh",
    "message",
    "metal",
    "method",
    "middle",
    "midnight",
    "milk",
    "million",
    "mimic",
    "mind",
    "minimum",
    "minor",
    "minute",
    "miracle",
    "mirror",
    "misery",
    "miss",
    "mistake",
    "mix",
    "mixed",
    "mixture",
    "mobile",
    "model",
    "modify",
    "mom",
    "moment",
    "monitor",
    "monkey",
    "monster",
    "month",
    "moon",
    "moral",
    "more",
    "morning",
    "mosquito",
    "mother",
    "motion",
    "motor",
    "mountain",
    "mouse",
    "move",
    "movie",
    "much",
    "muffin",
    "mule",
    "multiply",
    "muscle",
    "museum",
    "mushroom",
    "music",
    "must",
    "mutual",
    "myself",
    "mystery",
    "myth",
    "naive",
    "name",
    "napkin",
    "narrow",
    "nasty",
    "nation",
    "nature",
    "near",
    "neck",
    "need",
    "negative",
    "neglect",
    "neither",
    "nephew",
    "nerve",
    "nest",
    "net",
    "network",
    "neutral",
    "never",
    "news",
    "next",
    "nice",
    "night",
    "noble",
    "noise",
    "nominee",
    "noodle",
    "normal",
    "north",
    "nose",
    "notable",
    "note",
    "nothing",
    "notice",
    "novel",
    "now",
    "nuclear",
    "number",
    "nurse",
    "nut",
    "oak",
    "obey",
    "object",
    "oblige",
    "obscure",
    "observe",
    "obtain",
    "obvious",
    "occur",
    "ocean",
    "october",
    "odor",
    "off",
    "offer",
    "office",
    "often",
    "oil",
    "okay",
    "old",
    "olive",
    "olympic",
    "omit",
    "once",
    "one",
    "onion",
    "online",
    "only",
    "open",
    "opera",
    "opinion",
    "oppose",
    "option",
    "orange",
    "orbit",
    "orchard",
    "order",
    "ordinary",
    "organ",
    "orient",
    "original",
    "orphan",
    "ostrich",
    "other",
    "outdoor",
    "outer",
    "output",
    "outside",
    "oval",
    "oven",
    "over",
    "own",
    "owner",
    "oxygen",
    "oyster",
    "ozone",
    "pact",
    "paddle",
    "page",
    "pair",
    "palace",
    "palm",
    "panda",
    "panel",
    "panic",
    "panther",
    "paper",
    "parade",
    "parent",
    "park",
    "parrot",
    "party",
    "pass",
    "patch",
    "path",
    "patient",
    "patrol",
    "pattern",
    "pause",
    "pave",
    "payment",
    "peace",
    "peanut",
    "pear",
    "peasant",
    "pelican",
    "pen",
    "penalty",
    "pencil",
    "people",
    "pepper",
    "perfect",
    "permit",
    "person",
    "pet",
    "phone",
    "photo",
    "phrase",
    "physical",
    "piano",
    "picnic",
    "picture",
    "piece",
    "pig",
    "pigeon",
    "pill",
    "pilot",
    "pink",
    "pioneer",
    "pipe",
    "pistol",
    "pitch",
    "pizza",
    "place",
    "planet",
    "plastic",
    "plate",
    "play",
    "please",
    "pledge",
    "pluck",
    "plug",
    "plunge",
    "poem",
    "poet",
    "point",
    "polar",
    "pole",
    "police",
    "pond",
    "pony",
    "pool",
    "popular",
    "portion",
    "position",
    "possible",
    "post",
    "potato",
    "pottery",
    "poverty",
    "powder",
    "power",
    "practice",
    "praise",
    "predict",
    "prefer",
    "prepare",
    "present",
    "pretty",
    "prevent",
    "price",
    "pride",
    "primary",
    "print",
    "priority",
    "prison",
    "private",
    "prize",
    "problem",
    "process",
    "produce",
    "profit",
    "program",
    "project",
    "promote",
    "proof",
    "property",
    "prosper",
    "protect",
    "proud",
    "provide",
    "public",
    "pudding",
    "pull",
    "pulp",
    "pulse",
    "pumpkin",
    "punch",
    "pupil",
    "puppy",
    "purchase",
    "purity",
    "purpose",
    "purse",
    "push",
    "put",
    "puzzle",
    "pyramid",
    "quality",
    "quantum",
    "quarter",
    "question",
    "quick",
    "quit",
    "quiz",
    "quote",
    "rabbit",
    "raccoon",
    "race",
    "rack",
    "radar",
    "radio",
    "rail",
    "rain",
    "raise",
    "rally",
    "ramp",
    "ranch",
    "random",
    "range",
    "rapid",
    "rare",
    "rate",
    "rather",
    "raven",
    "raw",
    "razor",
    "ready",
    "real",
    "reason",
    "rebel",
    "rebuild",
    "recall",
    "receive",
    "recipe",
    "record",
    "recycle",
    "reduce",
    "reflect",
    "reform",
    "refuse",
    "region",
    "regret",
    "regular",
    "reject",
    "relax",
    "release",
    "relief",
    "rely",
    "remain",
    "remember",
    "remind",
    "remove",
    "render",
    "renew",
    "rent",
    "reopen",
    "repair",
    "repeat",
    "replace",
    "report",
    "require",
    "rescue",
    "resemble",
    "resist",
    "resource",
    "response",
    "result",
    "retire",
    "retreat",
    "return",
    "reunion",
    "reveal",
    "review",
    "reward",
    "rhythm",
    "rib",
    "ribbon",
    "rice",
    "rich",
    "ride",
    "ridge",
    "rifle",
    "right",
    "rigid",
    "ring",
    "riot",
    "ripple",
    "risk",
    "ritual",
    "rival",
    "river",
    "road",
    "roast",
    "robot",
    "robust",
    "rocket",
    "romance",
    "roof",
    "rookie",
    "room",
    "rose",
    "rotate",
    "rough",
    "round",
    "route",
    "royal",
    "rubber",
    "rude",
    "rug",
    "rule",
    "run",
    "runway",
    "rural",
    "sad",
    "saddle",
    "sadness",
    "safe",
    "sail",
    "salad",
    "salmon",
    "salon",
    "salt",
    "salute",
    "same",
    "sample",
    "sand",
    "satisfy",
    "satoshi",
    "sauce",
    "sausage",
    "save",
    "say",
    "scale",
    "scan",
    "scare",
    "scatter",
    "scene",
    "scheme",
    "school",
    "science",
    "scissors",
    "scorpion",
    "scout",
    "scrap",
    "screen",
    "script",
    "scrub",
    "sea",
    "search",
    "season",
    "seat",
    "second",
    "secret",
    "section",
    "security",
    "seed",
    "seek",
    "segment",
    "select",
    "sell",
    "seminar",
    "senior",
    "sense",
    "sentence",
    "series",
    "service",
    "session",
    "settle",
    "setup",
    "seven",
    "shadow",
    "shaft",
    "shallow",
    "share",
    "shed",
    "shell",
    "sheriff",
    "shield",
    "shift",
    "shine",
    "ship",
    "shiver",
    "shock",
    "shoe",
    "shoot",
    "shop",
    "short",
    "shoulder",
    "shove",
    "shrimp",
    "shrug",
    "shuffle",
    "shy",
    "sibling",
    "sick",
    "side",
    "siege",
    "sight",
    "sign",
    "silent",
    "silk",
    "silly",
    "silver",
    "similar",
    "simple",
    "since",
    "sing",
    "siren",
    "sister",
    "situate",
    "six",
    "size",
    "skate",
    "sketch",
    "ski",
    "skill",
    "skin",
    "skirt",
    "skull",
    "slab",
    "slam",
    "sleep",
    "slender",
    "slice",
    "slide",
    "slight",
    "slim",
    "slogan",
    "slot",
    "slow",
    "slush",
    "small",
    "smart",
    "smile",
    "smoke",
    "smooth",
    "snack",
    "snake",
    "snap",
    "sniff",
    "snow",
    "soap",
    "soccer",
    "social",
    "sock",
    "soda",
    "soft",
    "solar",
    "soldier",
    "solid",
    "solution",
    "solve",
    "someone",
    "song",
    "soon",
    "sorry",
    "sort",
    "soul",
    "sound",
    "soup",
    "source",
    "south",
    "space",
    "spare",
    "spatial",
    "spawn",
    "speak",
    "special",
    "speed",
    "spell",
    "spend",
    "sphere",
    "spice",
    "spider",
    "spike",
    "spin",
    "spirit",
    "split",
    "spoil",
    "sponsor",
    "spoon",
    "sport",
    "spot",
    "spray",
    "spread",
    "spring",
    "spy",
    "square",
    "squeeze",
    "squirrel",
    "stable",
    "stadium",
    "staff",
    "stage",
    "stairs",
    "stamp",
    "stand",
    "start",
    "state",
    "stay",
    "steak",
    "steel",
    "stem",
    "step",
    "stereo",
    "stick",
    "still",
    "sting",
    "stock",
    "stomach",
    "stone",
    "stool",
    "story",
    "stove",
    "strategy",
    "street",
    "strike",
    "strong",
    "struggle",
    "student",
    "stuff",
    "stumble",
    "style",
    "subject",
    "submit",
    "subway",
    "success",
    "such",
    "sudden",
    "suffer",
    "sugar",
    "suggest",
    "suit",
    "summer",
    "sun",
    "sunny",
    "sunset",
    "super",
    "supply",
    "supreme",
    "sure",
    "surface",
    "surge",
    "surprise",
    "surround",
    "survey",
    "suspect",
    "sustain",
    "swallow",
    "swamp",
    "swap",
    "swarm",
    "swear",
    "sweet",
    "swift",
    "swim",
    "swing",
    "switch",
    "sword",
    "symbol",
    "symptom",
    "syrup",
    "system",
    "table",
    "tackle",
    "tag",
    "tail",
    "talent",
    "talk",
    "tank",
    "tape",
    "target",
    "task",
    "taste",
    "tattoo",
    "taxi",
    "teach",
    "team",
    "tell",
    "ten",
    "tenant",
    "tennis",
    "tent",
    "term",
    "test",
    "text",
    "thank",
    "that",
    "theme",
    "then",
    "theory",
    "there",
    "they",
    "thing",
    "this",
    "thought",
    "three",
    "thrive",
    "throw",
    "thumb",
    "thunder",
    "ticket",
    "tide",
    "tiger",
    "tilt",
    "timber",
    "time",
    "tiny",
    "tip",
    "tired",
    "tissue",
    "title",
    "toast",
    "tobacco",
    "today",
    "toddler",
    "toe",
    "together",
    "toilet",
    "token",
    "tomato",
    "tomorrow",
    "tone",
    "tongue",
    "tonight",
    "tool",
    "tooth",
    "top",
    "topic",
    "topple",
    "torch",
    "tornado",
    "tortoise",
    "toss",
    "total",
    "tourist",
    "toward",
    "tower",
    "town",
    "toy",
    "track",
    "trade",
    "traffic",
    "tragic",
    "train",
    "transfer",
    "trap",
    "trash",
    "travel",
    "tray",
    "treat",
    "tree",
    "trend",
    "trial",
    "tribe",
    "trick",
    "trigger",
    "trim",
    "trip",
    "trophy",
    "trouble",
    "truck",
    "true",
    "truly",
    "trumpet",
    "trust",
    "truth",
    "try",
    "tube",
    "tuition",
    "tumble",
    "tuna",
    "tunnel",
    "turkey",
    "turn",
    "turtle",
    "twelve",
    "twenty",
    "twice",
    "twin",
    "twist",
    "two",
    "type",
    "typical",
    "ugly",
    "umbrella",
    "unable",
    "unaware",
    "uncle",
    "uncover",
    "under",
    "undo",
    "unfair",
    "unfold",
    "unhappy",
    "uniform",
    "unique",
    "unit",
    "universe",
    "unknown",
    "unlock",
    "until",
    "unusual",
    "unveil",
    "update",
    "upgrade",
    "uphold",
    "upon",
    "upper",
    "upset",
    "urban",
    "urge",
    "usage",
    "use",
    "used",
    "useful",
    "useless",
    "usual",
    "utility",
    "vacant",
    "vacuum",
    "vague",
    "valid",
    "valley",
    "valve",
    "van",
    "vanish",
    "vapor",
    "various",
    "vast",
    "vault",
    "vehicle",
    "velvet",
    "vendor",
    "venture",
    "venue",
    "verb",
    "verify",
    "version",
    "very",
    "vessel",
    "veteran",
    "viable",
    "vibrant",
    "vicious",
    "victory",
    "video",
    "view",
    "village",
    "vintage",
    "violin",
    "virtual",
    "virus",
    "visa",
    "visit",
    "visual",
    "vital",
    "vivid",
    "vocal",
    "voice",
    "void",
    "volcano",
    "volume",
    "vote",
    "voyage",
    "wage",
    "wagon",
    "wait",
    "walk",
    "wall",
    "walnut",
    "want",
    "warfare",
    "warm",
    "warrior",
    "wash",
    "wasp",
    "waste",
    "water",
    "wave",
    "way",
    "wealth",
    "weapon",
    "wear",
    "weasel",
    "weather",
    "web",
    "wedding",
    "weekend",
    "weird",
    "welcome",
    "west",
    "wet",
    "whale",
    "what",
    "wheat",
    "wheel",
    "when",
    "where",
    "whip",
    "whisper",
    "wide",
    "width",
    "wife",
    "wild",
    "will",
    "win",
    "window",
    "wine",
    "wing",
    "wink",
    "winner",
    "winter",
    "wire",
    "wisdom",
    "wise",
    "wish",
    "witness",
    "wolf",
    "woman",
    "wonder",
    "wood",
    "wool",
    "word",
    "work",
    "world",
    "worry",
    "worth",
    "wrap",
    "wreck",
    "wrestle",
    "wrist",
    "write",
    "wrong",
    "yard",
    "year",
    "yellow",
    "you",
    "young",
    "youth",
    "zebra",
    "zero",
    "zone",
    "zoo"];
}

var Module = (function () {
  var _scriptDir = typeof document !== 'undefined' && document.currentScript ? document.currentScript.src : undefined;
  if (typeof __filename !== 'undefined') _scriptDir = _scriptDir || __filename;
  return (
    function (Module) {
      Module = Module || {};
      var e; e || (e = typeof Module !== 'undefined' ? Module : {}); var aa, ba; e.ready = new Promise(function (a, b) { aa = a; ba = b; }); var p = {}, u; for (u in e) e.hasOwnProperty(u) && (p[u] = e[u]); var ca = !1, v = !1, da = !1; ca = "object" === typeof window; v = "function" === typeof importScripts; da = "object" === typeof process && "object" === typeof process.versions && "string" === typeof process.versions.node; var w = "", ea, fa, ia, ja;
      if (da) w = v ? require("path").dirname(w) + "/" : __dirname + "/", ea = function (a, b) { var c = ka(a); if (c) return b ? c : c.toString(); ia || (ia = require("fs")); ja || (ja = require("path")); a = ja.normalize(a); return ia.readFileSync(a, b ? null : "utf8") }, fa = function (a) { a = ea(a, !0); a.buffer || (a = new Uint8Array(a)); assert(a.buffer); return a }, 1 < process.argv.length && process.argv[1].replace(/\\/g, "/"), process.argv.slice(2), process.on("uncaughtException", function (a) { throw a; }), process.on("unhandledRejection", y), e.inspect = function () { return "[Emscripten Module object]" };
      else if (ca || v) v ? w = self.location.href : document.currentScript && (w = document.currentScript.src), _scriptDir && (w = _scriptDir), 0 !== w.indexOf("blob:") ? w = w.substr(0, w.lastIndexOf("/") + 1) : w = "", ea = function (a) {
        try { var b = new XMLHttpRequest; b.open("GET", a, !1); b.send(null); return b.responseText } catch (f) {
          if (a = ka(a)) { b = []; for (var c = 0; c < a.length; c++) { var d = a[c]; 255 < d && (d &= 255); b.push(String.fromCharCode(d)); } return b.join("") } throw f;
        }
      }, v && (fa = function (a) { try { var b = new XMLHttpRequest; b.open("GET", a, !1); b.responseType = "arraybuffer"; b.send(null); return new Uint8Array(b.response) } catch (c) { if (a = ka(a)) return a; throw c; } }); var ma = e.print || console.log.bind(console), z = e.printErr || console.warn.bind(console); for (u in p) p.hasOwnProperty(u) && (e[u] = p[u]); p = null; var A; e.wasmBinary && (A = e.wasmBinary); e.noExitRuntime && (e.noExitRuntime); "object" !== typeof WebAssembly && y("no native wasm support detected");
      var B, C = new WebAssembly.Table({ initial: 48, maximum: 48, element: "anyfunc" }), na = !1; function assert(a, b) { a || y("Assertion failed: " + b); } var oa = "undefined" !== typeof TextDecoder ? new TextDecoder("utf8") : void 0;
      function D(a, b, c) { var d = b + c; for (c = b; a[c] && !(c >= d);)++c; if (16 < c - b && a.subarray && oa) return oa.decode(a.subarray(b, c)); for (d = ""; b < c;) { var f = a[b++]; if (f & 128) { var g = a[b++] & 63; if (192 == (f & 224)) d += String.fromCharCode((f & 31) << 6 | g); else { var k = a[b++] & 63; f = 224 == (f & 240) ? (f & 15) << 12 | g << 6 | k : (f & 7) << 18 | g << 12 | k << 6 | a[b++] & 63; 65536 > f ? d += String.fromCharCode(f) : (f -= 65536, d += String.fromCharCode(55296 | f >> 10, 56320 | f & 1023)); } } else d += String.fromCharCode(f); } return d }
      function pa(a, b, c) { var d = E; if (0 < c) { c = b + c - 1; for (var f = 0; f < a.length; ++f) { var g = a.charCodeAt(f); if (55296 <= g && 57343 >= g) { var k = a.charCodeAt(++f); g = 65536 + ((g & 1023) << 10) | k & 1023; } if (127 >= g) { if (b >= c) break; d[b++] = g; } else { if (2047 >= g) { if (b + 1 >= c) break; d[b++] = 192 | g >> 6; } else { if (65535 >= g) { if (b + 2 >= c) break; d[b++] = 224 | g >> 12; } else { if (b + 3 >= c) break; d[b++] = 240 | g >> 18; d[b++] = 128 | g >> 12 & 63; } d[b++] = 128 | g >> 6 & 63; } d[b++] = 128 | g & 63; } } d[b] = 0; } } var qa = "undefined" !== typeof TextDecoder ? new TextDecoder("utf-16le") : void 0;
      function ra(a, b) { var c = a >> 1; for (var d = c + b / 2; !(c >= d) && sa[c];)++c; c <<= 1; if (32 < c - a && qa) return qa.decode(E.subarray(a, c)); c = 0; for (d = ""; ;) { var f = F[a + 2 * c >> 1]; if (0 == f || c == b / 2) return d; ++c; d += String.fromCharCode(f); } } function ta(a, b, c) { void 0 === c && (c = 2147483647); if (2 > c) return 0; c -= 2; var d = b; c = c < 2 * a.length ? c / 2 : a.length; for (var f = 0; f < c; ++f)F[b >> 1] = a.charCodeAt(f), b += 2; F[b >> 1] = 0; return b - d } function ua(a) { return 2 * a.length }
      function wa(a, b) { for (var c = 0, d = ""; !(c >= b / 4);) { var f = G[a + 4 * c >> 2]; if (0 == f) break; ++c; 65536 <= f ? (f -= 65536, d += String.fromCharCode(55296 | f >> 10, 56320 | f & 1023)) : d += String.fromCharCode(f); } return d } function xa(a, b, c) { void 0 === c && (c = 2147483647); if (4 > c) return 0; var d = b; c = d + c - 4; for (var f = 0; f < a.length; ++f) { var g = a.charCodeAt(f); if (55296 <= g && 57343 >= g) { var k = a.charCodeAt(++f); g = 65536 + ((g & 1023) << 10) | k & 1023; } G[b >> 2] = g; b += 4; if (b + 4 > c) break } G[b >> 2] = 0; return b - d }
      function ya(a) { for (var b = 0, c = 0; c < a.length; ++c) { var d = a.charCodeAt(c); 55296 <= d && 57343 >= d && ++c; b += 4; } return b } var za, H, E, F, sa, G, I, Aa, Ba, Ca = e.INITIAL_MEMORY || 16777216; e.wasmMemory ? B = e.wasmMemory : B = new WebAssembly.Memory({ initial: Ca / 65536, maximum: Ca / 65536 }); B && (za = B.buffer); Ca = za.byteLength; var J = za; za = J; e.HEAP8 = H = new Int8Array(J); e.HEAP16 = F = new Int16Array(J); e.HEAP32 = G = new Int32Array(J); e.HEAPU8 = E = new Uint8Array(J); e.HEAPU16 = sa = new Uint16Array(J); e.HEAPU32 = I = new Uint32Array(J); e.HEAPF32 = Aa = new Float32Array(J);
      e.HEAPF64 = Ba = new Float64Array(J); var Da = [], Ea = [], Fa = [], Ga = []; function Ha() { var a = e.preRun.shift(); Da.unshift(a); } var K = 0, Ja = null; e.preloadedImages = {}; e.preloadedAudios = {}; function y(a) { if (e.onAbort) e.onAbort(a); z(a); na = !0; a = new WebAssembly.RuntimeError("abort(" + a + "). Build with -s ASSERTIONS=1 for more info."); ba(a); throw a; } function Ka(a) { var b = La; return String.prototype.startsWith ? a.startsWith(b) : 0 === a.indexOf(b) } var La = "data:application/octet-stream;base64,", L = "data:application/octet-stream;base64,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IAEtAAA6AAAgAUEBaiEBIAJBAWoiAiADRw0ACwsgAAsbACAAIAIgBCACKAIAIAFzIANzIAVBACAGEDgLUgAgACAAKQMAIAEpAwB8NwMAIAAgACkDCCABKQMIfDcDCCAAIAApAxAgASkDEHw3AxAgACAAKQMYIAEpAxh8NwMYIAAgACkDICABKQMgfDcDIAtrAQJ+IAAgAkEBaq0iA0Le8P//3///D34gASkDAH03AwAgACADQv7///////8PfiIEIAEpAwh9NwMIIAAgBCABKQMQfTcDECAAIAQgASkDGH03AxggACADQv7///////8AfiABKQMgfTcDIAstACACRQRAIAAoAgQgASgCBEYPCyAAIAFGBEBBAQ8LIAAQggEgARCCARCsAkULIAAgACACIAQgASACKAIAIAMQigEgBUHO+s/KeiAGEDgLIAAgACACIAQgASACKAIAIAMQiQEgBUHp7bXTByAGEDgLIAAgACACIAQgASACKAIAIAMQiAEgBUHc+e74eCAGEDgLIAAgACACIAQgASACKAIAIAMQhgEgBUHz/cDrBiAGEDgLIAAgACACIAQgASACKAIAIAMQhgEgBUGh1+f2BiAGEDgLIAAgACACIAQgASACKAIAIAMQiAEgBUGkorfiBSAGEDgLIAAgACACIAQgASACKAIAIAMQiQEgBUGZ84nUBSAGEDgLIAAgACACIAQgASACKAIAIAMQigEgBUHml4qFBSAGEDgLGgAgACABIAIQlQEgAEEgaiABQSBqIAIQlQELCQAgACgAABBcC0gBAX4gACABrSICIAApAwB+NwMAIAAgACkDCCACfjcDCCAAIAApAxAgAn43AxAgACAAKQMYIAJ+NwMYIAAgACkDICACfjcDIAuWAQECfiAAIAApAwAgACkDICICQjCIQtGHgIAQfnwiAUL/////////B4M3AwAgACAAKQMIIAFCNIh8IgFC/////////weDNwMIIAAgACkDECABQjSIfCIBQv////////8HgzcDECAAIAApAxggAUI0iHwiAUL/////////B4M3AxggACACQv///////z+DIAFCNIh8NwMgC/MCAgJ/AX4CQCACRQ0AIAAgAmoiA0F/aiABOgAAIAAgAToAACACQQNJDQAgA0F+aiABOgAAIAAgAToAASADQX1qIAE6AAAgACABOgAC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AAuvBQECfyMAQcADayIEJAACQCABKAJ4BEAgACACEEsMAQsgAigCUARAIAMEQCADQQEQNgsgACABQYABEB8aDAELIABBADYCeCAEQZgDaiABQdAAaiIFEBYgBCABKQMgNwOQAyAEIAEpAxg3A4gDIAQgASkDEDcDgAMgBCABKQMINwP4AiAEIAEpAwA3A/ACIARB8AJqEC8gBEHIAmogAiAEQZgDahAZIAQgASkDSDcDwAIgBCABQUBrKQMANwO4AiAEIAEpAzg3A7ACIAQgASkDMDcDqAIgBCABKQMoNwOgAiAEQaACahAvIARB+AFqIAJBKGogBEGYA2oQGSAEQfgBaiAEQfgBaiAFEBkgBEHQAWogBEHwAmpBARAiIARB0AFqIARByAJqECEgBEGoAWogBEGgAmpBARAiIARBqAFqIARB+AFqECEgBEHQAWoQVARAIARBqAFqEFQEQCAAIAEgAxBVDAILIAMEQCADQQAQNgsgAEEBNgJ4DAELIARBgAFqIARBqAFqEBYgBEHYAGogBEHQAWoQFiAEQTBqIARB0AFqIARB2ABqEBkgAwRAIAMgBCkD0AE3AwAgAyAEKQPwATcDICADIAQpA+gBNwMYIAMgBCkD4AE3AxAgAyAEKQPYATcDCAsgAEHQAGogBSAEQdABahAZIARBCGogBEHwAmogBEHYAGoQGSAAIAQpAyg3AyAgACAEKQMgNwMYIAAgBCkDGDcDECAAIAQpAxA3AwggACAEKQMINwMAIABBAhAuIAAgBEEwahAhIAAgAEEDECIgACAEQYABahAhIABBKGoiASAAQQUQIiABIARBCGoQISABIAEgBEGoAWoQGSAEQTBqIARBMGogBEGgAmoQGSAEQTBqIARBMGpBARAiIAEgBEEwahAhCyAEQcADaiQAC/ICACAAIAExAB88AAAgACABMwEePAABIAAgASkDGEIoiDwAAiAAIAE1Ahw8AAMgACABKQMYQhiIPAAEIAAgASkDGEIQiDwABSAAIAEpAxhCCIg8AAYgACABKQMYPAAHIAAgATEAFzwACCAAIAEzARY8AAkgACABKQMQQiiIPAAKIAAgATUCFDwACyAAIAEpAxBCGIg8AAwgACABKQMQQhCIPAANIAAgASkDEEIIiDwADiAAIAEpAxA8AA8gACABMQAPPAAQIAAgATMBDjwAESAA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AEEAQRkgAkEBdmsgAkEfRht0IQIgASgCACEBA0AgASIEKAIEQXhxIABGDQIgAkEddiEBIAJBAXQhAiAEIAFBBHFqIgdBEGooAgAiAQ0ACyAHIAM2AhAgAyAENgIYCyADIAM2AgwgAyADNgIIDAELIAQoAggiACADNgIMIAQgAzYCCCADQQA2AhggAyAENgIMIAMgADYCCAtBhMAEQYTABCgCAEF/aiIANgIAIAANAEGswwQhAwNAIAMoAgAiAEEIaiEDIAANAAtBhMAEQX82AgALC20BAX8jAEGAAmsiBSQAIARBgMAEcSACIANMckUEQCAFIAFB/wFxIAIgA2siAkGAAiACQYACSSIBGxAwGiABRQRAA0AgACAFQYACEE4gAkGAfmoiAkH/AUsNAAsLIAAgBSACEE4LIAVBgAJqJAALMwEBfyAAQQEgABshAAJAA0AgABBYIgENAUHgvwQoAgAiAQRAIAERBQAMAQsLEAQACyABCwYAIAAQRAvZAQEDfyAAIAAoAmAiAyACajYCYAJAQcAAIANBP3EiBGsiBSACSw0AIABBIGoiAyAEaiABIAUQHxogACADEJYBIAEgBWohAUEAIQQgAiAFayICQcAASQ0AA0AgAyABKQAANwAAIAMgASkAODcAOCADIAEpADA3ADAgAyABKQAoNwAoIAMgASkAIDcAICADIAEpABg3ABggAyABKQAQNwAQIAMgASkACDcACCAAIAMQlgEgAUFAayEBIAJBQGoiAkE/Sw0ACwsgAgRAIAAgBGpBIGogASACEB8aCwt/AQJ+IABCACACrCIDfSIEIAEpAwCDIANCf3wiAyAAKQMAg4Q3AwAgACABKQMIIASDIAApAwggA4OENwMIIAAgASkDECAEgyAAKQMQIAODhDcDECAAIAEpAxggBIMgACkDGCADg4Q3AxggACABKQMgIASDIAApAyAgA4OENwMgC4sDAQV+IAAgATEAHyABMQAeQgiGhCABMQAdQhCGhCABMQAcQhiGhCABMQAbQiCGhCABMQAaQiiGhCABMQAZQg+DQjCGhCICNwMAIAAgAS0AGUEEdq0gATEAGEIEhoQgATEAF0IMhoQgATEAFkIUhoQgATEAFUIchoQgATEAFEIkhoQgATEAE0IshoQiAzcDCCAAIAExABIgATEAEUIIhoQgATEAEEIQhoQgATEAD0IYhoQgATEADkIghoQgATEADUIohoQgATEADEIPg0IwhoQiBDcDECAAIAEtAAxBBHatIAExAAtCBIaEIAExAApCDIaEIAExAAlCFIaEIAExAAhCHIaEIAExAAdCJIaEIAExAAZCLIaEIgU3AxggACABMQAFIAExAARCCIaEIAExAANCEIaEIAExAAJCGIaEIAExAAFCIIaEIAExAABCKIaEIgY3AyACfyAGQv///////z9SIAJCr/j//+///wdUckUEQEEAIAMgBIMgBYNC/////////wdRDQEaC0EBCwuAAQAgACABKAJQNgJ4IAAgASkDADcDACAAIAEpAwg3AwggACABKQMQNwMQIAAgASkDGDcDGCAAIAEpAyA3AyAgACABKQMoNwMoIAAgASkDMDcDMCAAIAEpAzg3AzggAEFAayABQUBrKQMANwMAIAAgASkDSDcDSCAAQdAAakEBEDYLBwAgAEEARwvFAQEDfyABIAJqIQQCQCAAKAJgQT9xIgNFDQAgAiADakHAAEkEQCADIQUMAQsgAEEgaiICIANqIAFBwAAgA2siAxAfGiAAIAApA2AgA618NwNgIAAgAkEBEJ4BIAEgA2ohAQsCfyAEIAFrIgJBwABOBEAgACABIAJBBnYQngEgACAAKQNgIAJBQHEiAq18NwNgIAEgAmohAQsgBCABSwsEQCAAIAVqQSBqIAEgBCABayIBEB8aIAAgACkDYCABrHw3A2ALIAALFwAgAC0AAEEgcUUEQCABIAIgABCZAgsLpQECAn8BfiABBEAgABDHAQNAIAJBCEYEQCABIABBwAAQHxoFIAAgAkEDdGoiAyADKQMAIgRCOIYgBEIohkKAgICAgIDA/wCDhCAEQhiGQoCAgICA4D+DIARCCIZCgICAgPAfg4SEIARCCIhCgICA+A+DIARCGIhCgID8B4OEIARCKIhCgP4DgyAEQjiIhISENwMAIAJBAWohAgwBCwsLIABB0AEQNQsJACAAIAE2AAALDwAgABA/KAIAIAAoAgBrCxAAIAAoAgAgASgCAEZBAXMLCgAgACgCBBCNAQvPAQEFfiAAKQMAIAApAyAiA0IwiELRh4CAEH58IgJC/////////weDIgFC0IeAgBCFIQQCfyABUEUEQEEAIARC/////////wdSDQEaCyABIAApAwggAkI0iHwiAUL/////////B4OEIAApAxAgAUI0iHwiAkL/////////B4OEIAApAxggAkI0iHwiBUL/////////B4OEIANC////////P4MgBUI0iHwiA4RQIAEgBIMgAoMgBYMgA0KAgICAgIDAB4WDQv////////8HUXILC2sAAkAgASgCeARAIABBATYCeCACRQ0BIAJBARA2DwsgAgRAIAIgASkDKDcDACACIAEpA0g3AyAgAiABQUBrKQMANwMYIAIgASkDODcDECACIAEpAzA3AwggAhAvIAJBAhAuCyAAIAEQ8AELC8UCAQd+IAApAwggACkDACAAKQMgIgRCMIhC0YeAgBB+fCIBQjSIfCICQv////////8HgyEFIAApAxggACkDECACQjSIfCIHQjSIfCIDQv////////8HgyEGIAAgBEL///////8/gyADQjSIfCIEQjCIIAIgB0L/////////B4MiAoMgA4NC/////////wdRIARC////////P1FxIAFC/////////weDIgNCrvj//+///wdWca2EUAR+IAQFIANC0YeAgBB8IgFC/////////weDIQMgBSABQjSIfCIBQv////////8HgyEFIAIgAUI0iHwiAUL/////////B4MhAiAGIAFCNIh8IgFC/////////weDIQYgAUI0iCAEfEL///////8/gws3AyAgACAGNwMYIAAgAjcDECAAIAU3AwggACADNwMACx0AIAAgARCcASAAQShqIAFBIGoQnAEgAEEANgJQC4suAQx/IwBBEGsiDCQAAkACQAJAAkACQAJAAkACQAJAAkACQAJAIABB9AFNBEBB5L8EKAIAIgVBECAAQQtqQXhxIABBC0kbIgZBA3YiAHYiAUEDcQRAIAFBf3NBAXEgAGoiAkEDdCIDQZTABGooAgAiAUEIaiEAAkAgASgCCCIGIANBjMAEaiIDRgRAQeS/BCAFQX4gAndxNgIADAELIAYgAzYCDCADIAY2AggLIAEgAkEDdCICQQNyNgIEIAEgAmoiASABKAIEQQFyNgIEDA0LIAZB7L8EKAIAIghNDQEgAQRAAkBBAiAAdCICQQAgAmtyIAEgAHRxIgBBACAAa3FBf2oiACAAQQx2QRBxIgB2IgFBBXZBCHEiAiAAciABIAJ2IgBBAnZBBHEiAXIgACABdiIAQQF2QQJxIgFyIAAg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ACAAKAIEQQFyNgIEDAELIAEgBkEDcjYCBCALIANBAXI2AgQgAyALaiADNgIAIAgEQCAIQQN2IgZBA3RBjMAEaiEAQfi/BCgCACECAn9BASAGdCIGIAVxRQRAQeS/BCAFIAZyNgIAIAAMAQsgACgCCAshBiAAIAI2AgggBiACNgIMIAIgADYCDCACIAY2AggLQfi/BCALNgIAQey/BCADNgIACyABQQhqIQALIAxBEGokACAACwoAIABBUGpBCkkLSwAgAEIANwNgIABCq7OP/JGjs/DbADcCGCAAQv+kuYjFkdqCm383AhAgAELy5rvjo6f9p6V/NwIIIABC58yn0NbQ67O7fzcCACAACyoAIAAEQCAAQcAOQcAAEB8iAEHQAGpBgAEQNSAAQgA3A0ggAEIANwNACwskACAAQQh0QYCA/AdxIABBGHRyIABBCHZBgP4DcSAAQRh2cnILGgEBfyAAKAIAIQEgACAAKAIAQQFqNgIAIAELLAEBfyMAQRBrIgEkACABIAAoAgQ2AgggAUEIahBrKAIAIQAgAUEQaiQAIAALsQECAX8BfiMAQRBrIgIkACACIAApA2BCA4YiA6cQXK1CIIYgA0IgiKcQXK2ENwAIIAEgAEGACEE3IAAoAmBrQT9xQQFqEE0gAkEIakEIEE0iACgCABA9IAFBBGogACgCBBA9IAFBCGogACgCCBA9IAFBDGogACgCDBA9IAFBEGogACgCEBA9IAFBFGogACgCFBA9IAFBGGogACgCGBA9IAFBHGogACgCHBA9IAJBEGokAAtXACAAIAEpAwAgASkDCEI0hoQ3AwAgACABKQMQQiiGIAEpAwhCDIiENwMIIAAgASkDGEIchiABKQMQQhiIhDcDECAAIAEpAyBCEIYgASkDGEIkiIQ3AxgLTQEBfyMAQdAAayIDJAAgA0EoaiACEBYgAyADQShqIAIQGSAAIAEgA0EoahAZIABBKGogAUEoaiADEBkgACABKAJ4NgJQIANB0ABqJAALkAEBBX4gACkDCCAAKQMAIAApAyAiAUIwiELRh4CAEH58IgJCNIh8IgMgAoQgACkDECADQjSIfCIEhCAAKQMYIARCNIh8IgWEQv////////8HgyABQv///////z+DIAVCNIh8IgGEUCADIAJC0IeAgBCFgyAEgyAFgyABQoCAgICAgMAHhYNC/////////wdRcgsmAEJ/IAKthkJ/hSAAIAFBA3ZB+P///wFxaikDACABQT9xrYiDpwunAgEFfiAAIAApAxggACkDECAAKQMIIAApAwAgACkDICIDQjCIQtGHgIAQfnwiAkI0iHwiAUI0iHwiBEI0iHwiBUI0iCADQv///////z+DfCIDQjCIIARC/////////weDIgQgAYMgBYNC/////////wdRIANC////////P1FxIAJC/////////weDIgJCrvj//+///wdWca2EQtGHgIAQfiACfCICQv////////8HgzcDACAAIAFC/////////weDIAJCNIh8IgFC/////////weDNwMIIAAgAUI0iCAEfCIBQv////////8HgzcDECAAIAVC/////////weDIAFCNIh8IgFC/////////weDNwMYIAAgAUI0iCADfEL///////8/gzcDIAseACAAIAEoAgBqIAIgAxAfGiABIAEoAgAgA2o2AgALngEBBX4gACABKQMAQn+FIgJCwoLZgc3Rl+m/f3wiA0IAQn8gARA0GyIEgzcDACAAIAEpAwhCf4UiBSADIAJUrXwiAkK7wKL66py317p/fCIDIASDNwMIIAAgASkDEEJ/hSIGIAIgBVStIAMgAlStfHwiAkJ+fCIDIASDNwMQIAAgAiAGVK0gAyACVK18IAEpAxhCf4V8Qn98IASDNwMYCwgAIACnQQFxC2oBAX8jAEFAaiICJAAgAiABEHcgACACKQM4NwA4IAAgAikDMDcAMCAAIAIpAyg3ACggACACKQMgNwAgIAAgAikDGDcAGCAAIAIpAxA3ABAgACACKQMINwAIIAAgAikDADcAACACQUBrJAALRgECfyAAKAIEIgVBCHUhBiAAKAIAIgAgASACKAIAIAZqKAIAIAYgBUEBcRsgAmogA0ECIAVBAnEbIAQgACgCACgCGBEIAAu8AQEDfwJ/IAEgACgCWEE/cSIERQ0AGiACIARqQcAASQRAIAQhBSABDAELIABBFGoiAyAEaiABQcAAIARrIgQQHxogACAAKQNYIAStfDcDWCAAIAMQiwEgASAEagshAyABIAJqIQIDQCACIANBQGsiAUlFBEAgACADEIsBIAAgACkDWEJAfTcDWCABIQMMAQsLIAIgA0sEQCAAIAVqQRRqIAMgAiADayIBEB8aIAAgACkDWCABrHw3A1gLIAALEQAgACAAKAIAQX9qNgIAIAALCgAgACgCABCNAQsfACAAQgA3AwggACABrTcDACAAQgA3AxAgAEIANwMYC8kBAQN/IwBB0ABrIgIkACAAIAEoAng2AlAgAUHQAGoiAyADEKIBIAJBKGogAxAWIAIgAyACQShqEBkgASABIAJBKGoQGSABQShqIgQgBCACEBkgA0EBEDYgACABKQMgNwMgIAAgASkDGDcDGCAAIAEpAxA3AxAgACABKQMINwMIIAAgASkDADcDACAAIAEpAyg3AyggACABKQMwNwMwIAAgASkDODcDOCAAQUBrIAFBQGspAwA3AwAgACABKQNINwNIIAJB0ABqJAALuwQBA38jAEHAAWsiAyQAIANCADcDMCADQgA3AzggA0FAa0IANwMAIANCADcDSCADQgA3A1AgA0IANwNYIANCADcDYCADQgA3AyggASAAQShqQYABEB8hBSADQQhqIAIgAEEIahCkASADQQA2ArgBA0AgA0EoaiAEQQp0IgEgACgCAGogA0EIaiAEQQJ0QQQQYyICRRAsIANBKGogACgCACABakFAayACQQFGECwgA0EoaiAAKAIAIAFqQYABaiACQQJGECwgA0EoaiAAKAIAIAFqQcABaiACQQNGECwgA0EoaiAAKAIAIAFqQYACaiACQQRGECwgA0EoaiAAKAIAIAFqQcACaiACQQVGECwgA0EoaiAAKAIAIAFqQYADaiACQQZGECwgA0EoaiAAKAIAIAFqQcADaiACQQdGECwgA0EoaiAAKAIAIAFqQYAEaiACQQhGECwgA0EoaiAAKAIAIAFqQcAEaiACQQlGECwgA0EoaiAAKAIAIAFqQYAFaiACQQpGECwgA0EoaiAAKAIAIAFqQcAFaiACQQtGECwgA0EoaiAAKAIAIAFqQYAGaiACQQxGECwgA0EoaiAAKAIAIAFqQcAGaiACQQ1GECwgA0EoaiAAKAIAIAFqQYAHaiACQQ5GECwgA0EoaiAAKAIAIAFqQcAHaiACQQ9GECwgA0HoAGogA0EoahBXIAUgBSADQegAahD/ASAEQQFqIgRBwABHDQALIANB6ABqEHggA0EIahBCIANBwAFqJAALuQEBAn8jAEHQAWsiAiQAIAAoAkAEQCACQQhqIABBIGoiAxBBIAJBCGogAEEgEDEgAkEIakHAtwRBARAxIAJBCGogAxBAIAJBCGogAxBBIAJBCGogAEEgEDEgAkEIaiAAEEALIAJBCGogAEEgahBBIAJBCGogAEEgEDEgAkEIaiAAEEAgASAAKQAYNwAYIAEgACkAEDcAECABIAApAAg3AAggASAAKQAANwAAIABBATYCQCACQdABaiQAC1ABAX8jAEGgGGsiBSQAIAUgBUEQajYCDCAFIAVBoAhqNgIIIAUgBUHgDWo2AgQgBSAFQaAQajYCACAAIAUgASACIAMgBBDpASAFQaAYaiQAC1IAIAAgAikAADcAACAAIAIpABg3ABggACACKQAQNwAQIAAgAikACDcACCABIAIpADg3ABggASACKQAwNwAQIAEgAikAKDcACCABIAIpACA3AAALhQEBBX4gACAAKQMAIgQgAa0iAkK//ab+sq7olsAAfnwiBTcDACAAIAApAwgiBiACQsS/3YWV48ioxQB+fCIDIAUgBFStfCIENwMIIAAgACkDECIFIAJ8IgIgAyAGVK0gBCADVK18fCIDNwMQIAAgACkDGCACIAVUrSADIAJUrXx8NwMYIAELZAICfwJ+IAApAxhCf1IgACkDECIDQn5UciIBIAApAwgiBEK7wKL66py317p/VHJBf3MiAiAEQrvAovrqnLfXun9WcSABQX9zIANCf1FxciAAKQMAQsCC2YHN0Zfpv39WIAJxcguiAwAgACABKQMgQiiIPAAAIAAgATUCJDwAASAAIAEpAyBCGIg8AAIgACABKQMgQhCIPAADIAAgASkDIEIIiDwABCAAIAEpAyA8AAUgACABKQMYQiyIPAAGIAAgASkDGEIkiDwAByAAIAEpAxhCHIg8AAggACABKQMYQhSIPAAJIAAgASkDGEIMiDwACiAAIAEpAxhCBIg8AAsgACABMwEWQg+DIAEpAxhCBIaEPAAMIAAgASkDEEIoiDwADSAAIAE1AhQ8AA4gACABKQMQQhiIPAAPIAAgASkDEEIQiDwAECAAIAEpAxBCCIg8ABEgACABKQMQPAASIAAgASkDCEIsiDwAEyAAIAEpAwhCJIg8ABQgACABKQMIQhyIPAAVIAAgASkDCEIUiDwAFiAAIAEpAwhCDIg8ABcgACABKQMIQgSIPAAYIAAgATMBBkIPgyABKQMIQgSGhDwAGSAAIAEpAwBCKIg8ABogACABNQIEPAAbIAAgASkDAEIYiDwAHCAAIAEpAwBCEIg8AB0gACABKQMAQgiIPAAeIAAgASkDADwAHwuOAQEBfyMAQUBqIgMkACADIAIpADg3AzggAyACKQAwNwMwIAMgAikAKDcDKCADIAIpACA3AyAgAyACKQAYNwMYIAMgAikAEDcDECADIAIpAAg3AwggAyACKQAANwMAIAEgAxBXQQEhAiABEK4BBEAgACgCsAEgACgCtAFBsbYEEB5BACECCyADQUBrJAAgAgudAQEBfyMAQdAAayICJAAgAiABKQMINwMwIAIgASkDEDcDOCACQUBrIAEpAxg3AwAgAiABKQMgNwNIIAIgASkDADcDKCACQShqEGQgAiABKQNINwMgIAIgAUFAaykDADcDGCACIAEpAzg3AxAgAiABKQMwNwMIIAIgASkDKDcDACACEGQgACACQShqEGAgAEEgaiACEGAgAkHQAGokAAsUACAAQQA2AlAgABA7IABBKGoQOwuQAQEDfyAAIQECQAJAIABBA3FFDQAgAC0AAEUEQEEADwsDQCABQQFqIgFBA3FFDQEgAS0AAA0ACwwBCwNAIAEiAkEEaiEBIAIoAgAiA0F/cyADQf/9+3dqcUGAgYKEeHFFDQALIANB/wFxRQRAIAIgAGsPCwNAIAItAAEhAyACQQFqIgEhAiADDQALCyABIABrC0gBAn8gACgCBCIGQQh1IQcgACgCACIAIAEgAiADKAIAIAdqKAIAIAcgBkEBcRsgA2ogBEECIAZBAnEbIAUgACgCACgCFBEJAAseACAAKAIcQQFGIAAoAgQgAUdyRQRAIAAgAjYCHAsLoQEAIABBAToANQJAIAAoAgQgAkcNACAAQQE6ADQgACgCECICRQRAIABBATYCJCAAIAM2AhggACABNgIQIANBAUcgACgCMEEBR3INASAAQQE6ADYPCyABIAJGBEAgACgCGCICQQJGBEAgACADNgIYIAMhAgsgACgCMEEBRyACQQFHcg0BIABBAToANg8LIABBAToANiAAIAAoAiRBAWo2AiQLC10BAX8gACgCECIDRQRAIABBATYCJCAAIAI2AhggACABNgIQDwsCQCABIANGBEAgACgCGEECRw0BIAAgAjYCGA8LIABBAToANiAAQQI2AhggACAAKAIkQQFqNgIkCwsUACAAQeApNgIA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IgIgBUF/c3IgA3NqQcOz7aoGakEGdyACaiIAaiACIA9qIAMgF2ogBSASaiAAIANBf3NyIAJzakGSmbP4eGpBCncgAGoiBSACQX9zciAAc2pB/ei/f2pBD3cgBWoiAiAAQX9zciAFc2pB0buRrHhqQRV3IAJqIgAgBUF/c3IgAnNqQc/8of0GakEGdyAAaiIDaiAAIA5qIAIgFWogBSAHaiADIAJBf3NyIABzakHgzbNxakEKdyADaiICIABBf3NyIANzakGUhoWYempBD3cgAmoiACADQX9zciACc2pBoaOg8ARqQRV3IABqIgUgAkF/c3IgAHNqQYL9zbp/akEGdyAFaiIDIBlqNgIAIAEgGCACIBZqIAMgAEF/c3IgBXNqQbXk6+l7akEKdyADaiICajYCDCABIBEgACATaiACIAVBf3NyIANzakG7pd/WAmpBD3cgAmoiAGo2AgggASAAIApqIAUgDWogACADQX9zciACc2pBkaeb3H5qQRV3ajYCBAsHACAAIAF3Cw0AIAFBf3MgAHIgAnMLJAEBfyMAQRBrIgIkACAAQQRBgAlBkAlBAyABEBUgAkEQaiQACxAAIAAgAnEgAkF/cyABcXILEAAgAEF/cyACcSAAIAFxcgsNACACQX9zIAFyIABzC6opARN/IwBBMGsiAiQAIAIgACgCACIDNgIsIAIgACgCBCISNgIoIAIgACgCCCIENgIkIAIgACgCDCITNgIgIAIgACgCECIUNgIcIAIgAzYCGCACIBI2AhQgAiAENgIQIAIgEzYCDCACIBQ2AgggASgABCEDIAEoAAghBCABKAAMIQUgASgAECEGIAEoABQhByABKAAYIQggASgAHCEJIAEoACAhCiABKAAkIQsgASgAKCEMIAEoACwhDSABKAAwIQ4gASgANCEPIAEoADghECABKAA8IREgAkEsaiASIAJBJGogEyAUIAEoAAAiAUELECAgAkEYaiACKAIUIAJBEGogAigCDCACKAIIIAdBCBArIAJBHGogAigCLCACQShqIAIoAiQgAigCICADQQ4QICACQQhqIAIoAhggAkEUaiACKAIQIAIoAgwgEEEJECsgAkEgaiACKAIcIAJBLGogAigCKCACKAIkIARBDxAgIAJBDGogAigCCCACQRhqIAIoAhQgAigCECAJQQkQKyACQSRqIAIoAiAgAkEcaiACKAIsIAIoAiggBUEMECAgAkEQaiACKAIMIAJBCGogAigCGCACKAIUIAFBCxArIAJBKGogAigCJCACQSBqIAIoAhwgAigCLCAGQQUQICACQRRqIAIoAhAgAkEMaiACKAIIIAIoAhggC0ENECsgAkEsaiACKAIoIAJBJGogAigCICACKAIcIAdBCBAgIAJBGGogAigCFCACQRBqIAIoAgwgAigCCCAEQQ8QKyACQRxqIAIoAiwgAkEoaiACKAIkIAIoAiAgCEEHECAgAkEIaiACKAIYIAJBFGogAigCECACKAIMIA1BDxArIAJBIGogAigCHCACQSxqIAIoAiggAigCJCAJQQkQICACQQxqIAIoAgggAkEYaiACKAIUIAIoAhAgBkEFECsgAkEkaiACKAIgIAJBHGogAigCLCACKAIoIApBCxAgIAJBEGogAigCDCACQQhqIAIoAhggAigCFCAPQQcQKyACQShqIAIoAiQgAkEgaiACKAIcIAIoAiwgC0ENECAgAkEUaiACKAIQIAJBDGogAigCCCACKAIYIAhBBxArIAJBLGogAigCKCACQSRqIAIoAiAgAigCHCAMQQ4QICACQRhqIAIoAhQgAkEQaiACKAIMIAIoAgggEUEIECsgAkEcaiACKAIsIAJBKGogAigCJCACKAIgIA1BDxAgIAJBCGogAigCGCACQRRqIAIoAhAgAigCDCAKQQsQKyACQSBqIAIoAhwgAkEsaiACKAIoIAIoAiQgDkEGECAgAkEMaiACKAIIIAJBGGogAigCFCACKAIQIANBDhArIAJBJGogAigCICACQRxqIAIoAiwgAigCKCAPQQcQICACQRBqIAIoAgwgAkEIaiACKAIYIAIoAhQgDEEOECsgAkEoaiACKAIkIAJBIGogAigCHCACKAIsIBBBCRAgIAJBFGogAigCECACQQxqIAIoAgggAigCGCAFQQwQKyACQSxqIAIoAiggAkEkaiACKAIgIAIoAhwgEUEIECAgAkEYaiACKAIUIAJBEGogAigCDCACKAIIIA5BBhArIAJBHGogAigCLCACQShqIAIoAiQgAigCICAJQQcQKiACQQhqIAIoAhggAkEUaiACKAIQIAIoAgwgCEEJECkgAkEgaiACKAIcIAJBLGogAigCKCACKAIkIAZBBhAqIAJBDGogAigCCCACQRhqIAIoAhQgAigCECANQQ0QKSACQSRqIAIoAiAgAkEcaiACKAIsIAIoAiggD0EIECogAkEQaiACKAIMIAJBCGogAigCGCACKAIUIAVBDxApIAJBKGogAigCJCACQSBqIAIoAhwgAigCLCADQQ0QKiACQRRqIAIoAhAgAkEMaiACKAIIIAIoAhggCUEHECkgAkEsaiACKAIoIAJBJGogAigCICACKAIcIAxBCxAqIAJBGGogAigCFCACQRBqIAIoAgwgAigCCCABQQwQKSACQRxqIAIoAiwgAkEoaiACKAIkIAIoAiAgCEEJECogAkEIaiACKAIYIAJBFGogAigCECACKAIMIA9BCBApIAJBIGogAigCHCACQSxqIAIoAiggAigCJCARQQcQKiACQQxqIAIoAgggAkEYaiACKAIUIAIoAhAgB0EJECkgAkEkaiACKAIgIAJBHGogAigCLCACKAIoIAVBDxAqIAJBEGogAigCDCACQQhqIAIoAhggAigCFCAMQQsQKSACQShqIAIoAiQgAkEgaiACKAIcIAIoAiwgDkEHECogAkEUaiACKAIQIAJBDGogAigCCCACKAIYIBBBBxApIAJBLGogAigCKCACQSRqIAIoAiAgAigCHCABQQwQKiACQRhqIAIoAhQgAkEQaiACKAIMIAIoAgggEUEHECkgAkEcaiACKAIsIAJBKGogAigCJCACKAIgIAtBDxAqIAJBCGogAigCGCACQRRqIAIoAhAgAigCDCAKQQwQKSACQSBqIAIoAhwgAkEsaiACKAIoIAIoAiQgB0EJECogAkEMaiACKAIIIAJBGGogAigCFCACKAIQIA5BBxApIAJBJGogAigCICACQRxqIAIoAiwgAigCKCAEQQsQKiACQRBqIAIoAgwgAkEIaiACKAIYIAIoAhQgBkEGECkgAkEoaiACKAIkIAJBIGogAigCHCACKAIsIBBBBxAqIAJBFGogAigCECACQQxqIAIoAgggAigCGCALQQ8QKSACQSxqIAIoAiggAkEkaiACKAIgIAIoAhwgDUENECogAkEYaiACKAIUIAJBEGogAigCDCACKAIIIANBDRApIAJBHGogAigCLCACQShqIAIoAiQgAigCICAKQQwQKiACQQhqIAIoAhggAkEUaiACKAIQIAIoAgwgBEELECkgAkEgaiACKAIcIAJBLGogAigCKCACKAIkIAVBCxAoIAJBDGogAigCCCACQRhqIAIoAhQgAigCECARQQkQJyACQSRqIAIoAiAgAkEcaiACKAIsIAIoAiggDEENECggAkEQaiACKAIMIAJBCGogAigCGCACKAIUIAdBBxAnIAJBKGogAigCJCACQSBqIAIoAhwgAigCLCAQQQYQKCACQRRqIAIoAhAgAkEMaiACKAIIIAIoAhggA0EPECcgAkEsaiACKAIoIAJBJGogAigCICACKAIcIAZBBxAoIAJBGGogAigCFCACQRBqIAIoAgwgAigCCCAFQQsQJyACQRxqIAIoAiwgAkEoaiACKAIkIAIoAiAgC0EOECggAkEIaiACKAIYIAJBFGogAigCECACKAIMIAlBCBAnIAJBIGogAigCHCACQSxqIAIoAiggAigCJCARQQkQKCACQQxqIAIoAgggAkEYaiACKAIUIAIoAhAgEEEGECcgAkEkaiACKAIgIAJBHGogAigCLCACKAIoIApBDRAoIAJBEGogAigCDCACQQhqIAIoAhggAigCFCAIQQYQJyACQShqIAIoAiQgAkEgaiACKAIcIAIoAiwgA0EPECggAkEUaiACKAIQIAJBDGogAigCCCACKAIYIAtBDhAnIAJBLGogAigCKCACQSRqIAIoAiAgAigCHCAEQQ4QKCACQRhqIAIoAhQgAkEQaiACKAIMIAIoAgggDUEMECcgAkEcaiACKAIsIAJBKGogAigCJCACKAIgIAlBCBAoIAJBCGogAigCGCACQRRqIAIoAhAgAigCDCAKQQ0QJyACQSBqIAIoAhwgAkEsaiACKAIoIAIoAiQgAUENECggAkEMaiACKAIIIAJBGGogAigCFCACKAIQIA5BBRAnIAJBJGogAigCICACQRxqIAIoAiwgAigCKCAIQQYQKCACQRBqIAIoAgwgAkEIaiACKAIYIAIoAhQgBEEOECcgAkEoaiACKAIkIAJBIGogAigCHCACKAIsIA9BBRAoIAJBFGogAigCECACQQxqIAIoAgggAigCGCAMQQ0QJyACQSxqIAIoAiggAkEkaiACKAIgIAIoAhwgDUEMECggAkEYaiACKAIUIAJBEGogAigCDCACKAIIIAFBDRAnIAJBHGogAigCLCACQShqIAIoAiQgAigCICAHQQcQKCACQQhqIAIoAhggAkEUaiACKAIQIAIoAgwgBkEHECcgAkEgaiACKAIcIAJBLGogAigCKCACKAIkIA5BBRAoIAJBDGogAigCCCACQRhqIAIoAhQgAigCECAPQQUQJyACQSRqIAIoAiAgAkEcaiACKAIsIAIoAiggA0ELECYgAkEQaiACKAIMIAJBCGogAigCGCACKAIUIApBDxAlIAJBKGogAigCJCACQSBqIAIoAhwgAigCLCALQQwQJiACQRRqIAIoAhAgAkEMaiACKAIIIAIoAhggCEEFECUgAkEsaiACKAIoIAJBJGogAigCICACKAIcIA1BDhAmIAJBGGogAigCFCACQRBqIAIoAgwgAigCCCAGQQgQJSACQRxqIAIoAiwgAkEoaiACKAIkIAIoAiAgDEEPECYgAkEIaiACKAIYIAJBFGogAigCECACKAIMIANBCxAlIAJBIGogAigCHCACQSxqIAIoAiggAigCJCABQQ4QJiACQQxqIAIoAgggAkEYaiACKAIUIAIoAhAgBUEOECUgAkEkaiACKAIgIAJBHGogAigCLCACKAIoIApBDxAmIAJBEGogAigCDCACQQhqIAIoAhggAigCFCANQQ4QJSACQShqIAIoAiQgAkEgaiACKAIcIAIoAiwgDkEJECYgAkEUaiACKAIQIAJBDGogAigCCCACKAIYIBFBBhAlIAJBLGogAigCKCACQSRqIAIoAiAgAigCHCAGQQgQJiACQRhqIAIoAhQgAkEQaiACKAIMIAIoAgggAUEOECUgAkEcaiACKAIsIAJBKGogAigCJCACKAIgIA9BCRAmIAJBCGogAigCGCACQRRqIAIoAhAgAigCDCAHQQYQJSACQSBqIAIoAhwgAkEsaiACKAIoIAIoAiQgBUEOECYgAkEMaiACKAIIIAJBGGogAigCFCACKAIQIA5BCRAlIAJBJGogAigCICACQRxqIAIoAiwgAigCKCAJQQUQJiACQRBqIAIoAgwgAkEIaiACKAIYIAIoAhQgBEEMECUgAkEoaiACKAIkIAJBIGogAigCHCACKAIsIBFBBhAmIAJBFGogAigCECACQQxqIAIoAgggAigCGCAPQQkQJSACQSxqIAIoAiggAkEkaiACKAIgIAIoAhwgEEEIECYgAkEYaiACKAIUIAJBEGogAigCDCACKAIIIAtBDBAlIAJBHGogAigCLCACQShqIAIoAiQgAigCICAHQQYQJiACQQhqIAIoAhggAkEUaiACKAIQIAIoAgwgCUEFECUgAkEgaiACKAIcIAJBLGogAigCKCACKAIkIAhBBRAmIAJBDGogAigCCCACQRhqIAIoAhQgAigCECAMQQ8QJSACQSRqIAIoAiAgAkEcaiACKAIsIAIoAiggBEEMECYgAkEQaiACKAIMIAJBCGogAigCGCACKAIUIBBBCBAlIAJBKGogAigCJCACQSBqIAIoAhwgAigCLCAGQQkQJCACQRRqIAIoAhAgAkEMaiACKAIIIAIoAhggDkEIECAgAkEsaiACKAIoIAJBJGogAigCICACKAIcIAFBDxAkIAJBGGogAigCFCACQRBqIAIoAgwgAigCCCARQQUQICACQRxqIAIoAiwgAkEoaiACKAIkIAIoAiAgB0EFECQgAkEIaiACKAIYIAJBFGogAigCECACKAIMIAxBDBAgIAJBIGogAigCHCACQSxqIAIoAiggAigCJCALQQsQJCACQQxqIAIoAgggAkEYaiACKAIUIAIoAhAgBkEJECAgAkEkaiACKAIgIAJBHGogAigCLCACKAIoIAlBBhAkIAJBEGogAigCDCACQQhqIAIoAhggAigCFCADQQwQICACQShqIAIoAiQgAkEgaiACKAIcIAIoAiwgDkEIECQgAkEUaiACKAIQIAJBDGogAigCCCACKAIYIAdBBRAgIAJBLGogAigCKCACQSRqIAIoAiAgAigCHCAEQQ0QJCACQRhqIAIoAhQgAkEQaiACKAIMIAIoAgggCkEOECAgAkEcaiACKAIsIAJBKGogAigCJCACKAIgIAxBDBAkIAJBCGogAigCGCACQRRqIAIoAhAgAigCDCAJQQYQICACQSBqIAIoAhwgAkEsaiACKAIoIAIoAiQgEEEFECQgAkEMaiACKAIIIAJBGGogAigCFCACKAIQIAhBCBAgIAJBJGogAigCICACQRxqIAIoAiwgAigCKCADQQwQJCACQRBqIAIoAgwgAkEIaiACKAIYIAIoAhQgBEENECAgAkEoaiACKAIkIAJBIGogAigCHCACKAIsIAVBDRAkIAJBFGogAigCECACQQxqIAIoAgggAigCGCAPQQYQICACQSxqIAIoAiggAkEkaiACKAIgIAIoAhwgCkEOECQgAkEYaiACKAIUIAJBEGogAigCDCACKAIIIBBBBRAgIAJBHGogAigCLCACQShqIAIoAiQgAigCICANQQsQJCACQQhqIAIoAhggAkEUaiACKAIQIAIoAgwgAUEPECAgAkEgaiACKAIcIAJBLGogAigCKCACKAIkIAhBCBAkIAJBDGogAigCCCACQRhqIAIoAhQgAigCECAFQQ0QICACQSRqIAIoAiAgAkEcaiACKAIsIAIoAiggEUEFECQgAkEQaiACKAIMIAJBCGogAigCGCACKAIUIAtBCxAgIAJBKGogAigCJCACQSBqIAIoAhwgAigCLCAPQQYQJCACQRRqIAIoAhAgAkEMaiACKAIIIAIoAhggDUELECAgACgCACEBIAAgAigCDCACKAIkIAAoAgRqajYCACAAIAIoAgggAigCICAAKAIIamo2AgQgACACKAIYIAIoAhwgACgCDGpqNgIIIAAgAigCFCACKAIsIAAoAhBqajYCDCAAIAIoAhAgASACKAIoamo2AhAgAkEwaiQACzgBA39BCBANIgIiAyIBQbQpNgIAIAFB4Ck2AgAgAUEEaiAAEL4CIANBkCo2AgAgAkGwKkEOEAwACyUBAX8jAEEQayIBJAAgAUEQaiQAIAFBCGoiASAANgIAIAEoAgALEAAgACABNgIEIAAgATYCAAs1AQF/IAAQURogABBRGiAAKAIABEAgACAAKAIAENQBIAAQPxogACgCACEBIAAQURogARBECwswAQF/IwBBEGsiAiQAIAIgACgCADYCCCACIAIoAgggAWo2AgggAkEQaiQAIAIoAggLNQEBfyMAQRBrIgIkACACIAAoAgQ2AgggAiABKAIENgIAIAJBCGogAhBSIQAgAkEQaiQAIAALCwAgACABEGwQjgELCwAgACABEFMQjgELHAAgABDaASABBEAgACABENkBIAAgARDYAQsgAAtpAQJ+IABCACACrCIDfSIEIAEpAwCDIANCf3wiAyAAKQMAg4Q3AwAgACABKQMIIASDIAApAwggA4OENwMIIAAgASkDECAEgyAAKQMQIAODhDcDECAAIAEpAxggBIMgACkDGCADg4Q3AxgLkz4BRX8gACABKAI4IgJBGHQgAkEIdEGAgPwHcXIgAkEIdkGA/gNxIAJBGHZyciICIAEoAjwiA0EYdCADQQh0QYCA/AdxciADQQh2QYD+A3EgA0EYdnJy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ECIKIAIpA7gBIAIpA8gBfHx8IAggAikDsAF8IgQgCFStfCAEIAIpA8ABfCIIIARUrXwiBCAKVK0gASkDGCILIAIpA5gBIAIpA6gBfHx8IAQgAikDkAF8IgogBFStfCAKIAIpA6ABfCIEIApUrXwgAyAEfCIKIARUrXwiAyALVK0gAikDeCACKQNgIgsgAikDiAF8fHwgAyACKQNwfCIEIANUrXwgBCACKQOAAXwiAyAEVK18IAMgBnwiBCADVK18IgZ8IgNCAEK//ab+sq7olsAAEBggAkEwaiADIAZUrSAGIAtUrSAFIAIpA2h8fHwiBkIAQr/9pv6yruiWwAAQGCACQUBrIANCAELEv92FlePIqMUAEBggAkEgaiAGQgBCxL/dhZXjyKjFABAYIAJBEGogCSACKQNQfCILIAlUrSAIIAIpA1h8fCIHIAhUrSACKQM4IAogAikDSHx8fCAHIAIpAzB8IgkgB1StfCAJIAIpA0B8IgggCVStfCIHIApUrSAEIAIpAyh8fCAHQr/9pv6yruiWwABCACAGIAVUIgEbfCIFIAdUrXwgBSACKQMgfCIHIAVUrXwgAyAHfCIJIAdUrXwiBSAEVCABaiAFQsS/3YWV48ioxQBCACABG3wiAyAFVGogAyAGfCIHIANUaq0iBUIAQr/9pv6yruiWwAAQGCAAIAIpAxAiBiALfCIENwMAIAIgBUIAQsS/3YWV48ioxQAQGCAAIAggAikDACIIfCIDIAIpAxggBCAGVK18fCIENwMIIAAgBSAJfCIGIAQgA1StIAIpAwggAyAIVK18fHwiAzcDECAAIAYgBVStIAMgBlStfCIFIAd8IgM3AxggACAAEHQgAyAFVGoQcxogAkHgAWokAAszAQF/IwBBMGsiAiQAIAJBCGogAEEBECIgAkEIaiABECEgAkEIahBUIQAgAkEwaiQAIAALgwEAIAAgASkDAEL/////////B4M3AwAgACABKQMIQgyGQoDg//////8HgyABKQMAQjSIhDcDCCAAIAEpAxBCGIZCgICA+P///weDIAEpAwhCKIiENwMQIAAgASkDGEIkhkKAgICAgP7/B4MgASkDEEIciIQ3AxggACABKQMYQhCINwMgCxQBAX9BBBBGIgEgACgCADYCACABC74hARF/IwBBIGsiAyQAA0AgAgRAIAMgACgCACIINgIcIAMgACgCBCIJNgIYIAMgACgCCCIHNgIUIAMgACgCDDYCECADIAAoAhAiBjYCDCADIAAoAhQiBDYCCCADIAAoAhgiBTYCBCADIAAoAhw2AgAgCCAJIAcgA0EQaiAGIAQgBSADIAEQLSIHQZjfqJQEahAaIAMoAgAgAygCHCADKAIYIANBFGogAygCECADKAIMIAMoAgggA0EEaiABQQRqEC0iBkGRid2JB2oQGiADKAIEIAMoAgAgAygCHCADQRhqIAMoAhQgAygCECADKAIMIANBCGogAUEIahAtIgRBz/eDrntqEBogAygCCCADKAIEIAMoAgAgA0EcaiADKAIYIAMoAhQgAygCECADQQxqIAFBDGoQLSIFQaW3181+ahAaIAMoAgwgAygCCCADKAIEIAMgAygCHCADKAIYIAMoAhQgA0EQaiABQRBqEC0iCkHbhNvKA2oQGiADKAIQIAMoAgwgAygCCCADQQRqIAMoAgAgAygCHCADKAIYIANBFGogAUEUahAtIgtB8aPEzwVqEBogAygCFCADKAIQIAMoAgwgA0EIaiADKAIEIAMoAgAgAygCHCADQRhqIAFBGGoQLSIMQaSF/pF5ahAaIAMoAhggAygCFCADKAIQIANBDGogAygCCCADKAIEIAMoAgAgA0EcaiABQRxqEC0iDUHVvfHYemoQGiADKAIcIAMoAhggAygCFCADQRBqIAMoAgwgAygCCCADKAIEIAMgAUEgahAtIhNBmNWewH1qEBogAygCACADKAIcIAMoAhggA0EUaiADKAIQIAMoAgwgAygCCCADQQRqIAFBJGoQLSIOQYG2jZQBahAaIAMoAgQgAygCACADKAIcIANBGGogAygCFCADKAIQIAMoAgwgA0EIaiABQShqEC0iD0G+i8ahAmoQGiADKAIIIAMoAgQgAygCACADQRxqIAMoAhggAygCFCADKAIQIANBDGogAUEsahAtIhBBw/uxqAVqEBogAygCDCADKAIIIAMoAgQgAyADKAIcIAMoAhggAygCFCADQRBqIAFBMGoQLSIRQfS6+ZUHahAaIAMoAhAgAygCDCADKAIIIANBBGogAygCACADKAIcIAMoAhggA0EUaiABQTRqEC0iEkH+4/qGeGoQGiADKAIUIAMoAhAgAygCDCADQQhqIAMoAgQgAygCACADKAIcIANBGGogAUE4ahAtIghBp43w3nlqEBogAygCGCADKAIUIAMoAhAgA0EMaiADKAIIIAMoAgQgAygCACADQRxqIAFBPGoQLSIJQfTi74x8ahAaIAMoAhwgAygCGCADKAIUIANBEGogAygCDCADKAIIIAMoAgQgAyAIEBwgByAOamogBhAbaiIHQcHT7aR+ahAaIAMoAgAgAygCHCADKAIYIANBFGogAygCECADKAIMIAMoAgggA0EEaiAJEBwgBiAPamogBBAbaiIGQYaP+f1+ahAaIAMoAgQgAygCACADKAIcIANBGGogAygCFCADKAIQIAMoAgwgA0EIaiAHEBwgBCAQamogBRAbaiIEQca7hv4AahAaIAMoAgggAygCBCADKAIAIANBHGogAygCGCADKAIUIAMoAhAgA0EMaiAGEBwgBSARamogChAbaiIFQczDsqACahAaIAMoAgwgAygCCCADKAIEIAMgAygCHCADKAIYIAMoAhQgA0EQaiAEEBwgCiASamogCxAbaiIKQe/YpO8CahAaIAMoAhAgAygCDCADKAIIIANBBGogAygCACADKAIcIAMoAhggA0EUaiAFEBwgCCALamogDBAbaiILQaqJ0tMEahAaIAMoAhQgAygCECADKAIMIANBCGogAygCBCADKAIAIAMoAhwgA0EYaiAKEBwgCSAMamogDRAbaiIMQdzTwuUFahAaIAMoAhggAygCFCADKAIQIANBDGogAygCCCADKAIEIAMoAgAgA0EcaiALEBwgByANamogExAbaiINQdqR5rcHahAaIAMoAhwgAygCGCADKAIUIANBEGogAygCDCADKAIIIAMoAgQgAyAMEBwgBiATamogDhAbaiITQdKi+cF5ahAaIAMoAgAgAygCHCADKAIYIANBFGogAygCECADKAIMIAMoAgggA0EEaiANEBwgBCAOamogDxAbaiIOQe2Mx8F6ahAaIAMoAgQgAygCACADKAIcIANBGGogAygCFCADKAIQIAMoAgwgA0EIaiATEBwgBSAPamogEBAbaiIPQcjPjIB7ahAaIAMoAgggAygCBCADKAIAIANBHGogAygCGCADKAIUIAMoAhAgA0EMaiAOEBwgCiAQamogERAbaiIQQcf/5fp7ahAaIAMoAgwgAygCCCADKAIEIAMgAygCHCADKAIYIAMoAhQgA0EQaiAPEBwgCyARamogEhAbaiIRQfOXgLd8ahAaIAMoAhAgAygCDCADKAIIIANBBGogAygCACADKAIcIAMoAhggA0EUaiAQEBwgDCASamogCBAbaiISQceinq19ahAaIAMoAhQgAygCECADKAIMIANBCGogAygCBCADKAIAIAMoAhwgA0EYaiAREBwgCCANamogCRAbaiIIQdHGqTZqEBogAygCGCADKAIUIAMoAhAgA0EMaiADKAIIIAMoAgQgAygCACADQRxqIBIQHCAJIBNqaiAHEBtqIglB59KkoQFqEBogAygCHCADKAIYIAMoAhQgA0EQaiADKAIMIAMoAgggAygCBCADIAgQHCAHIA5qaiAGEBtqIgdBhZXcvQJqEBogAygCACADKAIcIAMoAhggA0EUaiADKAIQIAMoAgwgAygCCCADQQRqIAkQHCAGIA9qaiAEEBtqIgZBuMLs8AJqEBogAygCBCADKAIAIAMoAhwgA0EYaiADKAIUIAMoAhAgAygCDCADQQhqIAcQHCAEIBBqaiAFEBtqIgRB/Nux6QRqEBogAygCCCADKAIEIAMoAgAgA0EcaiADKAIYIAMoAhQgAygCECADQQxqIAYQHCAFIBFqaiAKEBtqIgVBk5rgmQVqEBogAygCDCADKAIIIAMoAgQgAyADKAIcIAMoAhggAygCFCADQRBqIAQQHCAKIBJqaiALEBtqIgpB1OapqAZqEBogAygCECADKAIMIAMoAgggA0EEaiADKAIAIAMoAhwgAygCGCADQRRqIAUQHCAIIAtqaiAMEBtqIgtBu5WoswdqEBogAygCFCADKAIQIAMoAgwgA0EIaiADKAIEIAMoAgAgAygCHCADQRhqIAoQHCAJIAxqaiANEBtqIgxBrpKLjnhqEBogAygCGCADKAIUIAMoAhAgA0EMaiADKAIIIAMoAgQgAygCACADQRxqIAsQHCAHIA1qaiATEBtqIg1BhdnIk3lqEBogAygCHCADKAIYIAMoAhQgA0EQaiADKAIMIAMoAgggAygCBCADIAwQHCAGIBNqaiAOEBtqIhNBodH/lXpqEBogAygCACADKAIcIAMoAhggA0EUaiADKAIQIAMoAgwgAygCCCADQQRqIA0QHCAEIA5qaiAPEBtqIg5By8zpwHpqEBogAygCBCADKAIAIAMoAhwgA0EYaiADKAIUIAMoAhAgAygCDCADQQhqIBMQHCAFIA9qaiAQEBtqIg9B8JauknxqEBogAygCCCADKAIEIAMoAgAgA0EcaiADKAIYIAMoAhQgAygCECADQQxqIA4QHCAKIBBqaiAREBtqIhBBo6Oxu3xqEBogAygCDCADKAIIIAMoAgQgAyADKAIcIAMoAhggAygCFCADQRBqIA8QHCALIBFqaiASEBtqIhFBmdDLjH1qEBogAygCECADKAIMIAMoAgggA0EEaiADKAIAIAMoAhwgAygCGCADQRRqIBAQHCAMIBJqaiAIEBtqIhJBpIzktH1qEBogAygCFCADKAIQIAMoAgwgA0EIaiADKAIEIAMoAgAgAygCHCADQRhqIBEQHCAIIA1qaiAJEBtqIghBheu4oH9qEBogAygCGCADKAIUIAMoAhAgA0EMaiADKAIIIAMoAgQgAygCACADQRxqIBIQHCAJIBNqaiAHEBtqIglB8MCqgwFqEBogAygCHCADKAIYIAMoAhQgA0EQaiADKAIMIAMoAgggAygCBCADIAgQHCAHIA5qaiAGEBtqIgdBloKTzQFqEBogAygCACADKAIcIAMoAhggA0EUaiADKAIQIAMoAgwgAygCCCADQQRqIAkQHCAGIA9qaiAEEBtqIgZBiNjd8QFqEBogAygCBCADKAIAIAMoAhwgA0EYaiADKAIUIAMoAhAgAygCDCADQQhqIAcQHCAEIBBqaiAFEBtqIgRBzO6hugJqEBogAygCCCADKAIEIAMoAgAgA0EcaiADKAIYIAMoAhQgAygCECADQQxqIAYQHCAFIBFqaiAKEBtqIgVBtfnCpQNqEBogAygCDCADKAIIIAMoAgQgAyADKAIcIAMoAhggAygCFCADQRBqIAQQHCAKIBJqaiALEBtqIgpBs5nwyANqEBogAygCECADKAIMIAMoAgggA0EEaiADKAIAIAMoAhwgAygCGCADQRRqIAUQHCAIIAtqaiAMEBtqIgtBytTi9gRqEBogAygCFCADKAIQIAMoAgwgA0EIaiADKAIEIAMoAgAgAygCHCADQRhqIAoQHCAJIAxqaiANEBtqIgxBz5Tz3AVqEBogAygCGCADKAIUIAMoAhAgA0EMaiADKAIIIAMoAgQgAygCACADQRxqIAsQHCAHIA1qaiATEBtqIg1B89+5wQZqEBogAygCHCADKAIYIAMoAhQgA0EQaiADKAIMIAMoAgggAygCBCADIAwQHCAGIBNqaiAOEBtqIgZB7oW+pAdqEBogAygCACADKAIcIAMoAhggA0EUaiADKAIQIAMoAgwgAygCCCADQQRqIA0QHCAEIA5qaiAPEBtqIgRB78aVxQdqEBogAygCBCADKAIAIAMoAhwgA0EYaiADKAIUIAMoAhAgAygCDCADQQhqIAYQHCAFIA9qaiAQEBtqIgVBlPChpnhqEBogAygCCCADKAIEIAMoAgAgA0EcaiADKAIYIAMoAhQgAygCECADQQxqIAQQHCAKIBBqaiAREBtqIgRBiISc5nhqEBogAygCDCADKAIIIAMoAgQgAyADKAIcIAMoAhggAygCFCADQRBqIAUQHCALIBFqaiASEBtqIgVB+v/7hXlqEBogAygCECADKAIMIAMoAgggA0EEaiADKAIAIAMoAhwgAygCGCADQRRqIAQQHCAMIBJqaiAIEBtqIgRB69nBonpqEBogAygCFCADKAIQIAMoAgwgA0EIaiADKAIEIAMoAgAgAygCHCADQRhqIAUQHCAIIA1qaiAJEBtqQffH5vd7ahAaIAMoAhggAygCFCADKAIQIANBDGogAygCCCADKAIEIAMoAgAgA0EcaiAEEBwgBiAJamogBxAbakHy8cWzfGoQGiAAIAAoAgAgAygCHGo2AgAgACAAKAIEIAMoAhhqNgIEIAAgACgCCCADKAIUajYCCCAAIAAoAgwgAygCEGo2AgwgACAAKAIQIAMoAgxqNgIQIAAgACgCFCADKAIIajYCFCAAIAAoAhggAygCBGo2AhggACAAKAIcIAMoAgBqNgIcIAFBQGshASACQX9qIQIMAQsLIANBIGokAAsJACAAIAEQogELXAAgACADKQAANwAAIAAgAykAGDcAGCAAIAMpABA3ABAgACADKQAINwAIIAEgAykAODcAGCABIAMpADA3ABAgASADKQAoNwAIIAEgAykAIDcAACACIAMtAEA2AgALWQAgACABKQAANwAAIAAgASkAGDcAGCAAIAEpABA3ABAgACABKQAINwAIIAAgAikAADcAICAAIAIpAAg3ACggACACKQAQNwAwIAAgAikAGDcAOCAAIAM6AEALgBcBAn8jAEHgA2siAiQAIAJBuANqIAEQFiACQbgDaiACQbgDaiABEBkgAkGQA2ogAkG4A2oQFiACQZADaiACQZADaiABEBkgAiACKQOwAzcDiAMgAiACKQOoAzcDgAMgAiACKQOgAzcD+AIgAiACKQOYAzcD8AIgAiACKQOQAzcD6AIgAkHoAmogAkHoAmoQFiACQegCaiACQegCahAWIAJB6AJqIAJB6AJqEBYgAkHoAmogAkHoAmogAkGQA2oQGSACIAIpA4gDNwPgAiACIAIpA4ADNwPYAiACIAIpA/gCNwPQAiACIAIpA/ACNwPIAiACIAIpA+gCNwPAAiACQcACaiACQcACahAWIAJBwAJqIAJBwAJqEBYgAkHAAmogAkHAAmoQFiACQcACaiACQcACaiACQZADahAZIAIgAikD4AI3A7gCIAIgAikD2AI3A7ACIAIgAikD0AI3A6gCIAIgAikDyAI3A6ACIAIgAikDwAI3A5gCIAJBmAJqIAJBmAJqEBYgAkGYAmogAkGYAmoQFiACQZgCaiACQZgCaiACQbgDahAZIAIgAikDuAI3A5ACIAIgAikDsAI3A4gCIAIgAikDqAI3A4ACIAIgAikDoAI3A/gBIAIgAikDmAI3A/ABIAJB8AFqIAJB8AFqEBYgAkHwAWogAkHwAWoQFiACQfABaiACQfABahAWIAJB8AFqIAJB8AFqEBYgAkHwAWogAkHwAWoQFiACQfABaiACQfABahAWIAJB8AFqIAJB8AFqEBYgAkHwAWogAkHwAWoQFiACQfABaiACQfABahAWIAJB8AFqIAJB8AFqEBYgAkHwAWogAkHwAWoQFiACQfABaiACQfABaiACQZgCahAZIAIgAikDkAI3A+gBIAIgAikDiAI3A+ABIAIgAikDgAI3A9gBIAIgAikD+AE3A9ABIAIgAikD8AE3A8gBIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWogAkHwAWoQGSACIAIpA+gBNwPAASACIAIpA+ABNwO4ASACIAIpA9gBNwOwASACIAIpA9ABNwOoASACIAIpA8gBNwOgASACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWogAkHIAWoQGSACIAIpA8ABNwOYASACIAIpA7gBNwOQASACIAIpA7ABNwOIASACIAIpA6gBNwOAASACIAIpA6ABNwN4A0AgAkH4AGogAkH4AGoQFiADQQFqIgNB2ABHDQALIAJB+ABqIAJB+ABqIAJBoAFqEBkgAiACKQOYATcDcCACIAIpA5ABNwNoIAIgAikDiAE3A2AgAiACKQOAATcDWCACIAIpA3g3A1AgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqIAJByAFqEBkgAiACKQNwNwNIIAJBQGsiAyACKQNoNwMAIAIgAikDYDcDOCACIAIpA1g3AzAgAiACKQNQNwMoIAJBKGogAkEoahAWIAJBKGogAkEoahAWIAJBKGogAkEoahAWIAJBKGogAkEoaiACQZADahAZIAIgAikDSDcDICACIAMpAwA3AxggAiACKQM4NwMQIAIgAikDMDcDCCACIAIpAyg3AwAgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACIAJB8AFqEBkgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACEBYgAiACIAEQGSACIAIQFiACIAIQFiACIAIQFiACIAIgAkG4A2oQGSACIAIQFiACIAIQFiAAIAEgAhAZIAJB4ANqJAALnSEBAX8jAEHAA2siAiQAIAJBgAFqIAEQFyACQaADaiACQYABaiABEB0gAkHgAGogAkGAAWogAkGgA2oQHSACQYADaiACQeAAaiACQYABahAdIAJBQGsgAkGAA2ogAkGAAWoQHSACQSBqIAJBQGsgAkGAAWoQHSACIAJBIGogAkGAAWoQHSACQeACaiACEBcgAkHgAmogAkHgAmoQFyACQeACaiACQeACaiACQSBqEB0gAkHAAmogAkHgAmoQFyACQcACaiACQcACahAXIAJBwAJqIAJBwAJqIAJBoANqEB0gAkGgAmogAkHAAmoQFyACQaACaiACQaACahAXIAJBoAJqIAJBoAJqEBcgAkGgAmogAkGgAmoQFyACQaACaiACQaACahAXIAJBoAJqIAJBoAJqEBcgAkGgAmogAkGgAmogAkHgAmoQHSACQYACaiACQaACahAXIAJBgAJqIAJBgAJqEBcgAkGAAmogAkGAAmoQFyACQYACaiACQYACahAXIAJBgAJqIAJBgAJqEBcgAkGAAmogAkGAAmoQFyACQYACaiACQYACahAXIAJBgAJqIAJBgAJqEBcgAkGAAmogAkGAAmoQFyACQYACaiACQYACahAXIAJBgAJqIAJBgAJqEBcgAkGAAmogAkGAAmoQFyACQYACaiACQYACahAXIAJBgAJqIAJBgAJqEBcgAkGAAmogAkGAAmogAkGgAmoQHSACQeABaiACQYACahAXIAJB4AFqIAJB4AFqEBcgAkHgAWogAkHgAWoQFyACQeABaiACQeABahAXIAJB4AFqIAJB4AFqEBcgAkHgAWogAkHgAWoQFyACQeABaiACQeABahAXIAJB4AFqIAJB4AFqEBcgAkHgAWogAkHgAWoQFyACQeABaiACQeABahAXIAJB4AFqIAJB4AFqEBcgAkHgAWogAkHgAWoQFyACQeABaiACQeABahAXIAJB4AFqIAJB4AFqEBcgAkHgAWogAkHgAWoQFyACQeABaiACQeABahAXIAJB4AFqIAJB4AFqEBcgAkHgAWogAkHgAWoQFyACQeABaiACQeABahAXIAJB4AFqIAJB4AFqEBcgAkHgAWogAkHgAWoQFyACQeABaiACQeABahAXIAJB4AFqIAJB4AFqEBcgAkHgAWogAkHgAWoQFyACQeABaiACQeABahAXIAJB4AFqIAJB4AFqEBcgAkHgAWogAkHgAWoQFyACQeABaiACQeABahAXIAJB4AFqIAJB4AFqIAJBgAJqEB0gAkHAAWogAkHgAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqEBcgAkHAAWogAkHAAWoQFyACQcABaiACQcABahAXIAJBwAFqIAJBwAFqIAJB4AFqEB0gAkGgAWogAkHAAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqIAJBoAJqEB0gAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWogAkHgAGoQHSACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqIAJBgANqEB0gAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABaiACQeAAahAdIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABaiACQSBqEB0gAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABaiACQSBqEB0gAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABaiACQYADahAdIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABaiACQYADahAdIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqIAIQHSACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqIAJB4ABqEB0gAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWogAkGAA2oQHSACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWogAkFAaxAdIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqIAJB4ABqEB0gAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABaiACQYADahAdIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWogAkGAA2oQHSACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABaiACQcACahAdIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABaiACQUBrEB0gAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWogAkEgahAdIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWogAhAdIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABaiACQaADahAdIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqIAIQHSACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqIAIQHSACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqIAJBQGsQHSACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABaiABEB0gAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAJBoAFqIAJBoAFqEBcgAkGgAWogAkGgAWoQFyACQaABaiACQaABahAXIAAgAkGgAWogAkHgAmoQHSACQcADaiQAC5UBAQV+IAAgAikDACIDIAEpAwB8IgQ3AwAgACAEIANUrSIFIAEpAwh8IgMgAikDCHwiBjcDCCAAIAIpAxAiByABKQMQfCIEIAMgBVStIAYgA1StfHwiBTcDECAAIAIpAxgiBiABKQMYfCIDIAQgB1StIAUgBFStfHwiBDcDGCAAIAAQdCADIAZUrSAEIANUrXynahBzGgsMACAAQQBBxAAQMBoLngIBAn8jAEHQAWsiAyQAIABCgYKEiJCgwIABNwIAIABCADcCICAAQoGChIiQoMCAATcCGCAAQoGChIiQoMCAATcCECAAQoGChIiQoMCAATcCCCAAQgA3AiggAEIANwIwIABCADcCOCADQQhqIABBIGoiBBBBIANBCGogAEEgEDEgA0EIakH4tgRBARAxIANBCGogASACEDEgA0EIaiAEEEAgA0EIaiAEEEEgA0EIaiAAQSAQMSADQQhqIAAQQCADQQhqIAQQQSADQQhqIABBIBAxIANBCGpB+bYEQQEQMSADQQhqIAEgAhAxIANBCGogBBBAIANBCGogBBBBIANBCGogAEEgEDEgA0EIaiAAEEAgAEEANgJAIANB0AFqJAALiwECAX8BfgJAAkAgACkDGFBFIAApAyBCAFJyDQAgACkDECICQqOilQpWDQBBfyEBIAJCo6KVClINASAAKQMIIgJCgojxr7eh5QBWDQAgAkKCiPGvt6HlAFINAUEBIQEgACkDACICQu71pv6irugGVg0BQX9BACACQu71pv6irugGUhsPC0EBIQELIAELdQEBfyMAQdAAayICJAAgAkEoaiABQdAAahAWIAJBKGogAkEoaiAAEBkgAiABKQMgNwMgIAIgASkDGDcDGCACIAEpAxA3AxAgAiABKQMINwMIIAIgASkDADcDACACEC8gAkEoaiACEJsBIQAgAkHQAGokACAACwQAIAALCQAgACABEKMBC3ICAn8BfiAAKQMYIgNCP4inIgJBf3MiASAAKQMQQn9ScSADQv///////////wBUciAAKQMIIgNCnaCRvbXO26vdAFQgAXFyQX9zIgEgA0KdoJG9tc7bq90AVnEgAnIgASAAKQMAQqDB7MDm6Mv0X1ZxcgveAgEFfyMAQUBqIgMkACADQQA2AjwgA0IANwMoIANCADcDICADQgA3AxggA0IANwMQAkAgASgCACIEIAJGIAQtAABBAkdyDQAgASAEQQFqNgIAIANBDGogASACEK0BRQ0AIAMoAgwiBEUNACABKAIAIgUgBGogAksNAAJAAkAgBSwAACICIARBAkkiB3JFBEAgBSwAAUF/Sg0DQQEhBgwBCwJ/AkAgByACQX9HckUEQCAFLAABQQBODQEMBQtBASACQX9KDQEaCyADQQE2AjwgBS0AACECQQALIQYgAkH/AXENAQsgAyAEQX9qIgQ2AgwgASAFQQFqIgU2AgALAkACQCAEQSFPBEAgA0EBNgI8DAELIAZFDQAgAyAEa0EwaiAFIAQQHxogACADQRBqIANBPGoQMiADKAI8RQ0BCyAAQQAQbQsgASABKAIAIARqNgIAQQEhBgsgA0FAayQAIAYL4AEBBX8gAEEANgIAAkAgASgCACIGIAJPDQAgASAGQQFqIgQ2AgAgBi0AACIDQf8BRg0AIANBgAFxRQRAIAAgAzYCAEEBIQUMAQsgA0GAAUYNACADQf8AcSIHIAIgBGsiA0sgB0EES3INACAELQAAIgRFDQACQCAHRQRAQQAhBAwBCyAAIAQ2AgAgASAGQQJqIgU2AgAgB0F/aiIDBEADQCAAIAUtAAAgBEEIdHIiBDYCACABIAVBAWoiBTYCACADQX9qIgMNAAsLIAIgBWshAwsgBCADTSAEQf8AS3EPCyAFCyAAIAApAyAgACkDGCAAKQMQIAApAwggACkDAISEhIRQCzAAIAAgARDtAUUEQEEADwsgAEEoaiIBEFYgACkDKBBnIAJHBEAgASABQQEQIgtBAQscACAAQQA2AnggABA7IABBKGoQOyAAQdAAahA7C+gEAQN/IwBBgANrIgIkACACQgA3AzggAkIANwMwIAJCADcDKEEgIQMgAkIANwMgIAJCADcDGCACQgA3AxAgAkIANwMIIAJCADcDACABRQRAIABBKGoiBEHwtAQQSyAEIAQQ+QEgAEEIakEBEG0LIAJBkAFqIABBCGoQOiACIAIpA6gBNwMYIAIgAikDoAE3AxAgAiACKQOYATcDCCACIAIpA5ABNwMAIAJByABqIAIgAQR/IAIgASkAGDcDOCACIAEpABA3AzAgAiABKQAINwMoIAIgASkAADcDIEHAAAVBIAsQpgEgAkIANwM4IAJCADcDMCACQgA3AyggAkIANwMgIAJCADcDGCACQgA3AxAgAkIANwMIIAJCADcDAANAIAJByABqIAJBkAFqEHAgAiACQbgBaiACQZABahBKIgFFNgJEIAEEQCACIAJBuAFqEK4BIgFBAEc2AkQgAQ0BIABBKGoiASACQbgBahD4ASACQbgBahA7A0AgAkHIAGogAkGQAWoQcCACQeACaiACQZABaiACQcQAahAyIAIoAkQEQCACQQE2AkQMAQUgAiACQeACahA0IgNBAEc2AkQgAw0BIAJByABqEKUBIAJCADcDqAEgAkIANwOgASACQgA3A5gBIAJCADcDkAEgACACQeABaiACQeACahBvIAJB4AJqIAJB4AJqEGYgACACKQP4AjcDICAAIAIpA/ACNwMYIAAgAikD6AI3AxAgACACKQPgAjcDCCABIAJB4AFqQYABEB8aIAJB4AJqEEIgAkHgAWoQsAEgAkGAA2okAAsLBSACQQE2AkQMAQsLCwkAIABBADYCAAsZACAAIAAoAgAiACABQQ9qQXBxajYCACAACygAIABB/wFxQQFHBEBBLUEAQYAxEB5BAA8LQcCBIEHAASAAQYACcRsLRwACQCABRQ0AIAFB+CwQMyIBRSABKAIIIAAoAghBf3Nxcg0AIAAoAgwgASgCDEEAECNFDQAgACgCECABKAIQQQAQIw8LQQALUgEBfyAAKAIEIQQgACgCACIAIAECf0EAIAJFDQAaIARBCHUiASAEQQFxRQ0AGiACKAIAIAFqKAIACyACaiADQQIgBEECcRsgACgCACgCHBEHAAsDAAELJAEBfyMAQRBrIgMkACADIAI2AgwgACABIAIQugIgA0EQaiQAC5gCAAJAAkAgAUEUSw0AAkACQAJAAkACQAJAAkACQCABQXdqDgoAAQIJAwQFBgkHCAsgAiACKAIAIgFBBGo2AgAgACABKAIANgIADwsgAiACKAIAIgFBBGo2AgAgACABNAIANwMADwsgAiACKAIAIgFBBGo2AgAgACABNQIANwMADwsgAiACKAIAIgFBBGo2AgAgACABMgEANwMADwsgAiACKAIAIgFBBGo2AgAgACABMwEANwMADwsgAiACKAIAIgFBBGo2AgAgACABMAAANwMADwsgAiACKAIAIgFBBGo2AgAgACABMQAANwMADwsgACACQQARAAALDwsgAiACKAIAQQdqQXhxIgFBCGo2AgAgACABKQMANwMAC0IBA38gACgCACwAABBZBEADQCAAKAIAIgIsAAAhAyAAIAJBAWo2AgAgAyAB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ACgCBCIBQQF1aiICIAAoAgAiACACKAIAaigCACAAIAFBAXEbEQEAIQBB6AAQRiAAQegAEB8LUgEBfyABQQZ2IgMgASACakF/akEGdkYEQCAAIAEgAhBjDwsgACADQQN0aiIAKQMIQcAAIAFBP3EiAWuthiAAKQMAIAGtiIRCfyACrYZCf4WDpwviBQECfyMAQeADayIEJAACQCACKAJQBEAgACABQYABEB8aDAELIAEoAngEQCAAQQA2AnggBEG4A2ogAxAWIARBkANqIARBuANqIAMQGSAAIAIgBEG4A2oQGSAAQShqIAJBKGogBEGQA2oQGSAAQdAAakEBEDYMAQsgAEEANgJ4IARBuANqIAFB0ABqIgUgAxAZIARBkANqIARBuANqEBYgBCABKQMgNwOIAyAEIAEpAxg3A4ADIAQgASkDEDcD+AIgBCABKQMINwPwAiAEIAEpAwA3A+gCIARB6AJqEC8gBEHAAmogAiAEQZADahAZIAQgASkDSDcDuAIgBCABQUBrKQMANwOwAiAEIAEpAzg3A6gCIAQgASkDMDcDoAIgBCABKQMoNwOYAiAEQZgCahAvIARB8AFqIAJBKGogBEGQA2oQGSAEQfABaiAEQfABaiAEQbgDahAZIARByAFqIARB6AJqQQEQIiAEQcgBaiAEQcACahAhIARBoAFqIARBmAJqQQEQIiAEQaABaiAEQfABahAhIARByAFqEFQEQCAEQaABahBUBEAgACABQQAQVQwCCyAAQQE2AngMAQsgBEH4AGogBEGgAWoQFiAEQdAAaiAEQcgBahAWIARBKGogBEHIAWogBEHQAGoQGSAAIAUpAyA3A3AgACAFKQMYNwNoIAAgBSkDEDcDYCAAIAUpAwg3A1ggACAFKQMANwNQIABB0ABqIgEgASAEQcgBahAZIAQgBEHoAmogBEHQAGoQGSAAIAQpAyA3AyAgACAEKQMYNwMYIAAgBCkDEDcDECAAIAQpAwg3AwggACAEKQMANwMAIABBAhAuIAAgBEEoahAhIAAgAEEDECIgACAEQfgAahAhIABBKGoiASAAQQUQIiABIAQQISABIAEgBEGgAWoQGSAEQShqIARBKGogBEGYAmoQGSAEQShqIARBKGpBARAiIAEgBEEoahAhCyAEQeADaiQAC8UCAQN/IwBBMGsiBiQAIABB6ARqIgUgAkGAB2oiBCkDADcDACAFIAQpAyA3AyAgBSAEKQMYNwMYIAUgBCkDEDcDECAFIAQpAwg3AwggBSAEKQMoNwMoIAUgBCkDMDcDMCAFIAQpAzg3AzggBUFAayAEQUBrKQMANwMAIAUgBCkDSDcDSCAFQShqEC8gASAEKQNwNwMgIAEgBCkDaDcDGCABIAQpA2A3AxAgASAEKQNYNwMIIAEgBCkDUDcDACAFQQA2AlAgBiADKQO4AjcDKCAGIAMpA7ACNwMgIAYgAykDqAI3AxggBiADKQOgAjcDECAGIAMpA5gCNwMIQQchAQNAIAFBB0cEQCAGQQhqIAZBCGogAyABQShsahAZCyAAIAFBf2oiAUHYAGxqIAIgAUEHdGogBkEIahBhIAENAAsgBkEwaiQAC6UEAgN/AX4jAEGwAmsiAyQAIANBsAFqIAJBABBVIAMgAykD0AE3AyAgAyADKQPIATcDGCADIAMpA8ABNwMQIAMgAykDuAE3AwggAyADKQPgATcDMCADIAMpA+gBNwM4IANBQGsgAykD8AE3AwAgAyADKQP4ATcDSCADIAMpA7ABNwMAIAMgAykD2AE3AyggA0EANgJQIANB2ABqIAIgA0GAAmoiBRBhIAAgAykDeDcDICAAIAMpA3A3AxggACADKQNoNwMQIAAgAykDYDcDCCAAIAMpA1g3AwAgACADKQOAATcDKCAAIAMpA4gBNwMwIAAgAykDkAE3AzggAEFAayADKQOYATcDACAAIAMpA6ABNwNIIAAgAikDWDcDWCAAIAIpA2A3A2AgACACKQNoNwNoIAAgAikDcDcDcCACKQNQIQYgAEEANgJ4IAAgBjcDUCABIAMpA6ACNwMgIAEgAykDmAI3AxggASADKQOQAjcDECABIAMpA4gCNwMIIAEgAykDgAI3AwAgAEGAAWoiAiAAIAMgAUEoahA5IABBgAJqIgQgAiADIAFB0ABqEDkgAEGAA2oiAiAEIAMgAUH4AGoQOSAAQYAEaiIEIAIgAyABQaABahA5IABBgAVqIgIgBCADIAFByAFqEDkgAEGABmoiBCACIAMgAUHwAWoQOSAAQYAHaiAEIAMgAUGYAmoQOSAAQdAHaiIAIAAgBRAZIANBsAJqJAAL+AUCA38NfiMAQaABayICJAAgAkGQAWogASkDACIFQgAgBRAYIAAgAikDkAE3AwAgAkGAAWogASkDCEIAIAEpAwAQGCAAIAIpA4ABIgZCAYYiBSACKQOYAXwiBzcDCCACQeAAaiABKQMQQgAgASkDABAYIAJB8ABqIAEpAwgiCEIAIAgQGCAAIAcgBVQiA60gAikDiAEiB0IBhiIIIAUgBlSthHwiBSACKQNgIglCAYYiCnwiBiACKQNwfCILNwMQIAJBQGsgASkDGEIAIAEpAwAQGCACQdAAaiABKQMQQgAgASkDCBAYIAAgAyAFUHGtIAggB1StfCIIIAYgBVQiA60gAikDaCIMQgGGIg0gCiAJVK2EfCIJfCIFIAIpA3ggCyAGVK18fCIGIAIpA0AiCkIBhiILfCIHIAIpA1AiDkIBhiIPfCIQNwMYIAJBIGogASkDGEIAIAEpAwgQGCACQTBqIAEpAxAiEUIAIBEQGCAAIAYgBVStIAUgCFStIAMgCVBxrSANIAxUrXx8fCIIIAcgBlQiA60gAikDSCIJQgGGIgwgCyAKVK2EfCIKfCIFIBAgB1QiBK0gAikDWCILQgGGIg0gDyAOVK2EfCIOfCIGIAIpAyAiD0IBhiIQfCIHIAIpAzB8IhE3AyAgAkEQaiABKQMYQgAgASkDEBAYIAAgBiAFVK0gBCAOUHGtIAUgCFStIAMgClBxrSANIAtUrSAMIAlUrXx8fHx8IgggByAGVCIDrSACKQMoIglCAYYiCiAQIA9UrYR8Igt8IgUgAikDOCARIAdUrXx8IgYgAikDECIHQgGGIgx8Ig03AyggAiABKQMYIg5CACAOEBggACAGIAVUrSAFIAhUrSADIAtQca0gCiAJVK18fHwiCCANIAZUIgGtIAIpAxgiBkIBhiIJIAwgB1SthHwiB3wiBSACKQMAfCIKNwMwIAAgCiAFVK0gBSAIVK0gASAHUHGtIAIpAwggCSAGVK18fHx8NwM4IAJBoAFqJAALDgAgASACIAAoAgARAAALnQQBBX8jAEGACWsiBiQAAn8gBBA0IAMoAnhyRQRAIAEoAgwiCEEANgKECCAIIARBBRCZASEEIAEoAgwgBDYCgAggASgCACABKAIEIAMgASgCDCgChAhBB3RqEOYBIAEoAgggBkGACGogASgCACABKAIEEOUBQQEhCiAEQQAgBEEAShsMAQsgBkGACGpBARA2QQALIQQCQCAFRQRAQQAhCAwBCyAGIAVBDxCZASIIIAQgCCAEShshBAsgAkEBNgJ4IAIQOyACQShqEDsgAkHQAGoQOyAEQQFOBEAgBkHQCGohBQNAIAIgAkEAEFUgBCIDQX9qIQQCQCAKRQ0AIAMgASgCDCIHKAKACEoNACAHIARBAnRqKAIAIgdFDQAgASgCCCEJAkAgB0EBTgRAIAZBqAhqIAkgB0F/akECbUHYAGxqQdgAEB8aDAELIAZBqAhqIAkgB0F/c0ECbUHYAGxqQdgAEB8aIAUgBUEBECILIAIgAiAGQagIakEAEDkLAkAgAyAISg0AIAYgBEECdGooAgAiB0UNACAAKAIAIQkCQCAHQQFOBEAgBkGoCGogCSAHQX9qQQJtQQZ0ahBXDAELIAZBqAhqIAkgB0F/c0ECbUEGdGoQVyAFIAVBARAiCyACIAIgBkGoCGogBkGACGoQ5AELIANBAUoNAAsLIAIoAnhFBEAgAkHQAGoiACAAIAZBgAhqEBkLIAZBgAlqJAALygYCAX8JfiMAQYACayIDJAAgA0HwAWogAikDAEIAIAEpAwAQGCAAIAMpA/ABNwMAIANB0AFqIAIpAwhCACABKQMAEBggA0HgAWogAikDAEIAIAEpAwgQGCAAIAMpA9ABIgUgAykD+AF8IgQgAykD4AF8IgY3AwggA0GgAWogAikDEEIAIAEpAwAQGCADQbABaiACKQMIQgAgASkDCBAYIANBwAFqIAIpAwBCACABKQMQEBggACADKQPYASAEIAVUrXwiByADKQPoASAGIARUrXx8IgQgAykDoAF8IgUgAykDsAF8IgYgAykDwAF8Igg3AxAgA0HgAGogAikDGEIAIAEpAwAQGCADQfAAaiACKQMQQgAgASkDCBAYIANBgAFqIAIpAwhCACABKQMQEBggA0GQAWogAikDAEIAIAEpAxgQGCAAIAMpA6gBIAUgBFStfCIKIAQgB1StfCIEIAMpA7gBIAYgBVStfHwiBSADKQPIASAIIAZUrXx8IgYgAykDYHwiByADKQNwfCIIIAMpA4ABfCIJIAMpA5ABfCILNwMYIANBMGogAikDGEIAIAEpAwgQGCADQUBrIAIpAxBCACABKQMQEBggA0HQAGogAikDCEIAIAEpAxgQGCAAIAMpA5gBIAsgCVStfCILIAYgBVStIAUgBFStIAQgClStfHwiCiADKQNoIAcgBlStfHwiBCADKQN4IAggB1StfHwiBSADKQOIASAJIAhUrXx8Igl8IgYgAykDMHwiByADKQNAfCIIIAMpA1B8Igw3AyAgA0EQaiACKQMYQgAgASkDEBAYIANBIGogAikDEEIAIAEpAxgQGCAAIAYgC1StIAkgBVStIAUgBFStIAQgClStfHx8IgkgAykDOCAHIAZUrXx8IgQgAykDSCAIIAdUrXx8IgUgAykDWCAMIAhUrXx8IgYgAykDEHwiByADKQMgfCIINwMoIAMgAikDGEIAIAEpAxgQGCAAIAYgBVStIAUgBFStIAQgCVStfHwiCSADKQMYIAcgBlStfHwiBCADKQMoIAggB1StfHwiBSADKQMAfCIGNwMwIAAgBiAFVK0gBSAEVK0gAykDCCAEIAlUrXx8fDcDOCADQYACaiQACzMBAX8jAEEwayICJAAgAkEIaiAAQQEQIiACQQhqIAEQISACQQhqEGIhACACQTBqJAAgAAv1FgECfyMAQeADayICJAAgAkG4A2ogARAWIAJBuANqIAJBuANqIAEQGSACQZADaiACQbgDahAWIAJBkANqIAJBkANqIAEQGSACIAIpA7ADNwOIAyACIAIpA6gDNwOAAyACIAIpA6ADNwP4AiACIAIpA5gDNwPwAiACIAIpA5ADNwPoAiACQegCaiACQegCahAWIAJB6AJqIAJB6AJqEBYgAkHoAmogAkHoAmoQFiACQegCaiACQegCaiACQZADahAZIAIgAikDiAM3A+ACIAIgAikDgAM3A9gCIAIgAikD+AI3A9ACIAIgAikD8AI3A8gCIAIgAikD6AI3A8ACIAJBwAJqIAJBwAJqEBYgAkHAAmogAkHAAmoQFiACQcACaiACQcACahAWIAJBwAJqIAJBwAJqIAJBkANqEBkgAiACKQPgAjcDuAIgAiACKQPYAjcDsAIgAiACKQPQAjcDqAIgAiACKQPIAjcDoAIgAiACKQPAAjcDmAIgAkGYAmogAkGYAmoQFiACQZgCaiACQZgCahAWIAJBmAJqIAJBmAJqIAJBuANqEBkgAiACKQO4AjcDkAIgAiACKQOwAjcDiAIgAiACKQOoAjcDgAIgAiACKQOgAjcD+AEgAiACKQOYAjcD8AEgAkHwAWogAkHwAWoQFiACQfABaiACQfABahAWIAJB8AFqIAJB8AFqEBYgAkHwAWogAkHwAWoQFiACQfABaiACQfABahAWIAJB8AFqIAJB8AFqEBYgAkHwAWogAkHwAWoQFiACQfABaiACQfABahAWIAJB8AFqIAJB8AFqEBYgAkHwAWogAkHwAWoQFiACQfABaiACQfABahAWIAJB8AFqIAJB8AFqIAJBmAJqEBkgAiACKQOQAjcD6AEgAiACKQOIAjcD4AEgAiACKQOAAjcD2AEgAiACKQP4ATcD0AEgAiACKQPwATcDyAEgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBahAWIAJByAFqIAJByAFqEBYgAkHIAWogAkHIAWoQFiACQcgBaiACQcgBaiACQfABahAZIAIgAikD6AE3A8ABIAIgAikD4AE3A7gBIAIgAikD2AE3A7ABIAIgAikD0AE3A6gBIAIgAikDyAE3A6ABIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABahAWIAJBoAFqIAJBoAFqEBYgAkGgAWogAkGgAWoQFiACQaABaiACQaABaiACQcgBahAZIAIgAikDwAE3A5gBIAIgAikDuAE3A5ABIAIgAikDsAE3A4gBIAIgAikDqAE3A4ABIAIgAikDoAE3A3gDQCACQfgAaiACQfgAahAWIANBAWoiA0HYAEcNAAsgAkH4AGogAkH4AGogAkGgAWoQGSACIAIpA5gBNwNwIAIgAikDkAE3A2ggAiACKQOIATcDYCACIAIpA4ABNwNYIAIgAikDeDcDUCACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGoQFiACQdAAaiACQdAAahAWIAJB0ABqIAJB0ABqEBYgAkHQAGogAkHQAGogAkHIAWoQGSACIAIpA3A3A0ggAkFAayIDIAIpA2g3AwAgAiACKQNgNwM4IAIgAikDWDcDMCACIAIpA1A3AyggAkEoaiACQShqEBYgAkEoaiACQShqEBYgAkEoaiACQShqEBYgAkEoaiACQShqIAJBkANqEBkgAiACKQNINwMgIAIgAykDADcDGCACIAIpAzg3AxAgAiACKQMwNwMIIAIgAikDKDcDACACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIgAkHwAWoQGSACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIQFiACIAIgAkG4A2oQGSACIAIQFiAAIAIQFiACIAAQFiACIAEQ6wEhACACQeADaiQAIAALjgEBAX8jAEGAAWsiAiQAIAAgASkDIDcDICAAIAEpAxg3AxggACABKQMQNwMQIAAgASkDCDcDCCAAIAEpAwA3AwAgAkHYAGogARAWIAJBMGogASACQdgAahAZIABBADYCUCACQQhqQQcQNiACQQhqIAJBMGoQISAAQShqIAJBCGoQ7AEhACACQYABaiQAIAAL4QgCAX8PfiMAQdACayICJAAgAkFAayABKQMYIgRCACABKQMAIgdCAYYiChAYIAJBkAJqIAEpAwgiCEIBhiIFQgAgASkDECIGEBggAkHgAWogASkDICIJQgAgCRAYIAJB0AFqIAIpA+ABIgNC/////////weDQgBCkPqAgIACEBggAkGwAWogCUIBhiIJQgAgBxAYIAJB0ABqIARCACAFEBggAkGAAmogBkIAIAYQGCACQcABaiACKQPoASIFQgyGIANCNIiEIAVCNIhCkPqAgIACEBggAkHAAmogB0IAIAcQGCACQaABaiAJQgAgCBAYIAJB4ABqIAZCAYZCACAEEBggAiACKQOgASIPIAIpA2B8IgUgAikDUCIOIAIpA4ACfCIDIAIpA7ABfCILIAIpA8ABfCIMIAIpA0AiDSACKQOQAnwiByACKQPQAXwiEEI0iCAQIAdUrSACKQPYASAHIA1UrSACKQNIIAIpA5gCfHx8fCINQgyGhHwiB0I0iCAHIAxUrSAMIAtUrSACKQPIASALIANUrSACKQO4ASADIA5UrSACKQNYIAIpA4gCfHx8fHx8IA1CNIh8fCIOQgyGhHwiA0IEhkLw/////////wCDIAdCMIhCD4OEQgBC0YeAgBAQGCAAIAIpAwAiDSACKQPAAnwiC0L/////////B4M3AwAgAkGwAmogCkIAIAgQGCACQZABaiAJQgAgBhAYIAJB8ABqIARCACAEEBggAkEwaiACKQOQASIRIAIpA3B8IgwgAyAFVK0gBSAPVK0gAikDqAEgAikDaHx8IA5CNIh8fCIPQgyGIANCNIiEfCIFQv////////8Hg0IAQpD6gICAAhAYIAAgAikDMCIOIAIpA7ACfCIDIAsgDVStIAIpAwggAikDyAJ8fCINQgyGIAtCNIiEfCILQv////////8HgzcDCCACQfABaiAGQgAgChAYIAJBoAJqIAhCACAIEBggAkGAAWogCUIAIAQQGCACQSBqIAUgDFStIAwgEVStIAIpA5gBIAIpA3h8fCAPQjSIfHwiCkIMhiAFQjSIhCIFIAIpA4ABfCIEQv////////8Hg0IAQpD6gICAAhAYIAAgAikD8AEiDCACKQOgAnwiBiACKQMgfCIIIAsgA1StIAMgDlStIAIpAzggAikDuAJ8fCANQjSIfHwiA0IMhiALQjSIhHwiCUL/////////B4M3AxAgAkEQaiAEIAVUrSACKQOIASAKQjSIfHwiCkIMhiAEQjSIhCAKQjSIQpD6gICAAhAYIAAgAikDECIKIBBC/v///////weDfCIEIAkgCFStIAggBlStIAIpAyggBiAMVK0gAikD+AEgAikDqAJ8fHx8IANCNIh8fCIIQgyGIAlCNIiEfCIGQv////////8HgzcDGCAAIAdC////////P4MgBiAEVK0gAikDGCAEIApUrXwgCEI0iHx8QgyGIAZCNIiEfDcDICACQdACaiQAC9oLAgF/GX4jAEHwA2siAyQAIANBQGsgAikDGCIEQgAgASkDACIFEBggA0HQAWogAikDECIGQgAgASkDCCILEBggA0HAAmogAikDCCIMQgAgASkDECINEBggA0GQA2ogAikDACIHQgAgASkDGCIPEBggA0HgA2ogAikDICIQQgAgASkDICIREBggA0HQA2ogAykD4AMiCEL/////////B4NCAEKQ+oCAgAIQGCADQdAAaiAQQgAgBRAYIANBkAFqIARCACALEBggA0GQAmogBkIAIA0QGCADQfACaiAMQgAgDxAYIANBsANqIAdCACAREBggA0HAA2ogAykD6AMiDkIMhiAIQjSIhCAOQjSIQpD6gICAAhAYIANB4ABqIAdCACAFEBggA0HgAWogEEIAIAsQGCADQaABaiAEQgAgDRAYIANBoAJqIAZCACAPEBggA0GAA2ogDEIAIBEQGCADIAMpA6ACIhkgAykDoAF8Ig4gAykDgAN8IhMgAykD4AF8IhQgAykDkAIiGyADKQOQAXwiCSADKQPwAnwiFSADKQOwA3wiFiADKQNQfCIXIAMpA8ADfCISIAMpA9ABIhwgAykDQHwiCCADKQPAAnwiCiADKQOQA3wiGCADKQPQA3wiGkI0iCAaIBhUrSADKQPYAyAYIApUrSADKQOYAyAKIAhUrSADKQPIAiAIIBxUrSADKQPYASADKQNIfHx8fHx8fHwiCkIMhoR8IghCNIggCCASVK0gEiAXVK0gAykDyAMgFyAWVK0gAykDWCAWIBVUrSADKQO4AyAVIAlUrSADKQP4AiAJIBtUrSADKQOYAiADKQOYAXx8fHx8fHx8fHwgCkI0iHx8IhJCDIaEfCIJQgSGQvD/////////AIMgCEIwiEIPg4RCAELRh4CAEBAYIAAgAykDACIKIAMpA2B8IhVC/////////weDNwMAIANB8ABqIAxCACAFEBggA0HwAWogB0IAIAsQGCADQdACaiAQQgAgDRAYIANBsAFqIARCACAPEBggA0GwAmogBkIAIBEQGCADQTBqIAMpA7ACIhggAykDsAF8IhYgAykD0AJ8IhcgCSAUVK0gFCATVK0gAykD6AEgEyAOVK0gAykDiAMgDiAZVK0gAykDqAIgAykDqAF8fHx8fHwgEkI0iHx8IhJCDIYgCUI0iIR8Ig5C/////////weDQgBCkPqAgIACEBggACADKQPwASIZIAMpA3B8IhMgAykDMHwiFCAVIApUrSADKQMIIAMpA2h8fCIKQgyGIBVCNIiEfCIJQv////////8HgzcDCCADQYABaiAGQgAgBRAYIANBgAJqIAxCACALEBggA0HgAmogB0IAIA0QGCADQaADaiAQQgAgDxAYIANBwAFqIARCACAREBggA0EgaiADKQOgAyIHIAMpA8ABfCIEIA4gF1StIBcgFlStIAMpA9gCIBYgGFStIAMpA7gCIAMpA7gBfHx8fCASQjSIfHwiD0IMhiAOQjSIhHwiBUL/////////B4NCAEKQ+oCAgAIQGCAAIAMpA4ACIhAgAykDgAF8IgYgAykD4AJ8IgsgAykDIHwiDCAJIBRUrSAUIBNUrSADKQM4IBMgGVStIAMpA/gBIAMpA3h8fHx8IApCNIh8fCIRQgyGIAlCNIiEfCINQv////////8HgzcDECADQRBqIAUgBFStIAQgB1StIAMpA6gDIAMpA8gBfHwgD0I0iHx8IgRCDIYgBUI0iIQgBEI0iEKQ+oCAgAIQGCAAIAMpAxAiByAaQv////////8Hg3wiBCANIAxUrSAMIAtUrSADKQMoIAsgBlStIAMpA+gCIAYgEFStIAMpA4gCIAMpA4gBfHx8fHx8IBFCNIh8fCIGQgyGIA1CNIiEfCIFQv////////8HgzcDGCAAIAhC////////P4MgBSAEVK0gAykDGCAEIAdUrXwgBkI0iHx8QgyGIAVCNIiEfDcDICADQfADaiQAC6gCAQN/IwBBoAFrIgIkACAAQQA2AnggAEHQAGoiAyABQdAAaiABQShqIgQQGSADQQIQLiACQfgAaiABEBYgAkH4AGpBAxAuIAJB0ABqIAJB+ABqEBYgAkEoaiAEEBYgAkEoakECEC4gAiACQShqEBYgAkECEC4gAkEoaiACQShqIAEQGSAAIAIpA0g3AyAgACACQUBrKQMANwMYIAAgAikDODcDECAAIAIpAzA3AwggACACKQMoNwMAIABBBBAuIAAgAEEEECIgACACQdAAahAhIAJB0ABqIAJB0ABqQQEQIiAC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ASkDMDcDMCAAIAEpAzg3AzggAEFAayABQUBrKQMANwMAIAAgASkDSDcDSCAAIAEpA1g3A1ggACABKQNgNwNgIAAgASkDaDcDaCAAIAEpA3A3A3AgACABKQNQNwNQIABBKGoiABAvIAAgAEEBECILBwAgABEOAAsYACAAKAIIEEwEQCAAQQhqIAEQsQELQQELVAEBfyMAQaABayIDJAAgA0EgaiABEEsgA0EBEG0gACADQSBqIANBIGogAyACEHFBACEAIAMoApgBRQRAIAEgA0EgahBuQQEhAAsgA0GgAWokACAAC6sCAQJ/IwBBgAFrIgMkACADQQA2AgQCQCAAKAIAEExFBEAgACgCsAEgACgCtAFBjzMQHgwBCyABRQRAIAAoArABIAAoArQBQZgxEB4MAQsgAkUEQCAAKAKwASAAKAK0AUGuNBAeDAELIANBCGogAiADQQRqEDIgAygCBARAIAFCADcAACABQgA3ADggAUIANwAwIAFCADcAKCABQgA3ACAgAUIANwAYIAFCADcAECABQgA3AAgMAQsgACADQShqIAEQdiECIAFCADcAOCABQgA3ADAgAUIANwAoIAFCADcAICABQgA3ABggAUIANwAQIAFCADcACCABQgA3AAAgAkUNACAAIANBKGogA0EIahD8AUUNACABIANBKGoQaEEBIQQLIANBgAFqJAAgBAs4AQF/IwBB8ABrIgMkACADQQhqEFogACABEE0gAhBfIANBCGoQWiACQSAQTSACEF8gA0HwAGokAAuMBwEGfyMAQZAEayIDJAAgA0HoA2ogAUHQAGoiBRAWIAMgASkDIDcD4AMgAyABKQMYNwPYAyADIAEpAxA3A9ADIAMgASkDCDcDyAMgAyABKQMANwPAAyADQcADahAvIANBmANqIAIgA0HoA2oQGSADIAEpA0g3A5ADIAMgAUFAaykDADcDiAMgAyABKQM4NwOAAyADIAEpAzA3A/gCIAMgASkDKDcD8AIgA0HwAmoQLyADQcgCaiACQShqIgcgA0HoA2oQGSADQcgCaiADQcgCaiAFEBkgAyADKQPgAzcDwAIgAyADKQPYAzcDuAIgAyADKQPQAzcDsAIgAyADKQPIAzcDqAIgAyADKQPAAzcDoAIgA0GgAmogA0GYA2oQISADIAMpA5ADNwPwASADIAMpA4gDNwPoASADIAMpA4ADNwPgASADIAMpA/gCNwPYASADIAMpA/ACNwPQASADQdABaiADQcgCahAhIANB2ABqIANBoAJqEBYgA0EwaiADQZgDakEBECIgA0H4AWogA0HAA2ogA0EwahAZIANB2ABqIANB+AFqECEgA0HQAWoQYiEEIANB2ABqEGIhBiADIAMpA5ADNwMoIAMgAykDiAM3AyAgAyADKQOAAzcDGCADIAMpA/gCNwMQIAMgAykD8AI3AwggA0EIakECEC4gA0EwaiADQcADahAhIANBCGogA0HYAGogBCAGcSIERSIGEEkgA0EwaiADQdABaiAGEEkgA0GoAWogA0EwahAWIANBgAFqIANBqAFqIANBoAJqEBkgA0GoAWogA0GoAWoQFiADQagBaiADQdABaiAEEEkgA0GgAmogA0EIahAWIABB0ABqIgQgBSADQTBqEBkgBBBiIQYgASgCeCEIIARBAhAuIANBgAFqIANBgAFqQQEQIiADQaACaiADQYABahAhIANBoAJqEC8gACADKQPAAjcDICAAIAMpA7gCNwMYIAAgAykDsAI3AxAgACADKQOoAjcDCCAAIAMpA6ACNwMAIANBoAJqQQIQLiADQaACaiADQYABahAhIANBoAJqIANBoAJqIANBCGoQGSADQaACaiADQagBahAhIABBKGoiBSADQaACakEDECIgBRAvIABBBBAuIAVBBBAuIAAgAiABKAJ4EEkgBSAHIAEoAngQSSAEQci3BCABKAJ4EEkgACAGQQEgCGtsNgJ4IANBkARqJAALhwIBAX8jAEGAAmsiAyQAAkAgAUUEQCAAKAKwASAAKAK0AUGYMRAeQQAhAgwBCyABQgA3AAAgAUIANwA4IAFCADcAMCABQgA3ACggAUIANwAgIAFCADcAGCABQgA3ABAgAUIANwAIIAAoAggQTEUEQCAAKAKwASAAKAK0AUHRMxAeQQAhAgwBCyACRQRAIAAoArABIAAoArQBQZ80EB5BACECDAELIANBCGogAiADQQRqEDJBACECAkAgAygCBA0AIANBCGoQNA0AIABBCGogA0GAAWogA0EIahBvIANBKGogA0GAAWoQbiABIANBKGoQaEEBIQILIANBCGoQQgsgA0GAAmokACACC/MBAQJ/IwBBoAJrIgckACAHQQA2AgQgACAHQYABaiAFEG8gB0Eo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1AFBA3RqIgQgBCkDACAAKQMAhTcDACAAQgA3AwAgAEEANgLQASAAIAAoAtQBQQFqIgQ2AtQBIAIgBmshAiAEQRkgACgC2AFB/////wdxa0cNACADEIMBIABBADYC1AELIABBCGohBCACQQN2IQZBACEDA0AgAyAGRgRAIAJBB3EhAwNAIANFDQMgATEAACEHIAAgACgC0AEiAkEBajYC0AEgACAAKQMAIAcgAkEDdK2GhDcDACABQQFqIQEgA0F/aiEDDAALAAsgBCAAKALUAUEDdGoiBSABKQAAIAUpAwCFNwMAIAAgACgC1AFBAWoiBTYC1AFBGSAAKALYAUH/////B3FrIAVGBEAgBBCDASAAQQA2AtQBCyABQQhqIQEgA0EBaiEDDAALAAsLNwEBfyMAQSBrIgMkACADENgCIAMgATYCFCADKAIQIAAgARA+IAMQxQEgAiADEMQBIANBIGokAAs9AQF/IABBgAgQWCIBNgIQIAFBgAgQwwEgAEIANwIUIABC/rnrxemOlZkQNwIIIABCgcaUupbx6uZvNwIACzEAIABBADYAPCAAIAFBFXY6ADsgACABQQ12OgA6IAAgAUEFdjoAOSAAIAFBA3Q6ADgLDQAgASACIAMgABECAAsfAEHACEEBEIcBQc8IQQIQhwEQ0gJBkL8EQQ8RAQAaCwu5rQQdAEGACAsBgABBwAgL5wFfc2luZ2xlX3NoYTI1NgBfZG91YmxlX3NoYTI1NgBDU0hBMjU2AFdyaXRlAEZpbmFsaXplAFJlc2V0AAAAAAAA2BYAADgXAABcFwAAOBcAAHZpaWlpADdDU0hBMjU2AACEFwAAlgQAAFA3Q1NIQTI1NgAAAGQYAACoBAAAAAAAAKAEAABQSzdDU0hBMjU2AABkGAAAxAQAAAEAAACgBAAAaWkAdgB2aQC0BAAAAAAAAKAEAACgBAAAOBcAAFwXAABpaWlpaQAAANgWAACgBAAAOBcAAHZpaWkAAAAAoAQAALQEAABpaWkAQbAKC8AD/////////////////////////////////////////////////////////////////wABAgMEBQYHCP////////8JCgsMDQ4PEP8REhMUFf8WFxgZGhscHR4fIP///////yEiIyQlJicoKSor/ywtLi8wMTIzNDU2Nzg5/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////2NhcnJ5ID09IDAAL2pzYmdsL2NwcC1jcnlwdG8vYmFzZTU4LmNwcABEZWNvZGVCYXNlNTgARW5jb2RlQmFzZTU4ADEyMzQ1Njc4OUFCQ0RFRkdISktMTU5QUVJTVFVWV1hZWmFiY2RlZmdoaWprbW5vcHFyc3R1dnd4eXoAYWxsb2NhdG9yPFQ+OjphbGxvY2F0ZShzaXplX3QgbikgJ24nIGV4Y2VlZHMgbWF4aW11bSBzdXBwb3J0ZWQgc2l6ZQBBgA4LAYAAQcAOC/IFCMm882fmCWo7p8qEha5nuyv4lP5y82488TYdXzr1T6XRguatf1IOUR9sPiuMaAWba71B+6vZgx95IX4TGc3gWyKuKNeYL4pCzWXvI5FEN3EvO03sz/vAtbzbiYGl27XpOLVI81vCVjkZ0AW28RHxWZtPGa+kgj+SGIFt2tVeHKtCAgOjmKoH2L5vcEUBW4MSjLLkTr6FMSTitP/Vw30MVW+Je/J0Xb5ysZYWO/6x3oA1Esclpwbcm5Qmac908ZvB0krxnsFpm+TjJU84hke+77XVjIvGncEPZZysd8yhDCR1AitZbyzpLYPkpm6qhHRK1PtBvdypsFy1UxGD2oj5dqvfZu5SUT6YEDK0LW3GMag/IfuYyCcDsOQO777Hf1m/wo+oPfML4MYlpwqTR5Gn1W+CA+BRY8oGcG4OCmcpKRT8L9JGhQq3JybJJlw4IRsu7SrEWvxtLE3fs5WdEw04U95jr4tUcwplqLJ3PLsKanbmru1HLsnCgTs1ghSFLHKSZAPxTKHov6IBMEK8S2YaqJGX+NBwi0vCML5UBqNRbMcYUu/WGeiS0RCpZVUkBpnWKiBxV4U1DvS40bsycKBqEMjQ0rgWwaQZU6tBUQhsNx6Z647fTHdIJ6hIm+G1vLA0Y1rJxbMMHDnLikHjSqrYTnPjY3dPypxbo7iy1vNvLmj8su9d7oKPdGAvF0NvY6V4cqvwoRR4yITsOWQaCALHjCgeYyP6/76Q6b2C3utsUKQVecay96P5vitTcuPyeHHGnGEm6s4+J8oHwsAhx7iG0R7r4M3WfdrqeNFu7n9PffW6bxdyqmfwBqaYyKLFfWMKrg35vgSYPxEbRxwTNQtxG4R9BCP1d9sokyTHQHuryjK8vskVCr6ePEwNEJzEZx1DtkI+y77UxUwqfmX8nCl/Wez61jqrb8tfF1hHSowZRGxjb250ZXh0LT5zaXplIDwgNjQAL2pzYmdsL2MtY3J5cHRvL21kNS5jAG1kNV9maW5hbABBwBQLsREKAAAABwAAAAsAAAARAAAAEgAAAAMAAAAFAAAAEAAAAAgAAAAVAAAAGAAAAAQAAAAPAAAAFwAAABMAAAANAAAADAAAAAIAAAAUAAAADgAAABYAAAAJAAAABgAAAAEAAAABAAAAAwAAAAYAAAAKAAAADwAAABUAAAAcAAAAJAAAAC0AAAA3AAAAAgAAAA4AAAAbAAAAKQAAADgAAAAIAAAAGQAAACsAAAA+AAAAEgAAACcAAAA9AAAAFAAAACwAAAABAAAAAAAAAIKAAAAAAAAAioAAAAAAAIAAgACAAAAAgIuAAAAAAAAAAQAAgAAAAACBgACAAAAAgAmAAAAAAACAigAAAAAAAACIAAAAAAAAAAmAAIAAAAAACgAAgAAAAACLgACAAAAAAIsAAAAAAACAiYAAAAAAAIADgAAAAAAAgAKAAAAAAACAgAAAAAAAAIAKgAAAAAAAAAoAAIAAAACAgYAAgAAAAICAgAAAAAAAgAEAAIAAAAAACIAAgAAAAIB2b2lkAGJvb2wAY2hhcgBzaWduZWQgY2hhcgB1bnNpZ25lZCBjaGFyAHNob3J0AHVuc2lnbmVkIHNob3J0AGludAB1bnNpZ25lZCBpbnQAbG9uZwB1bnNpZ25lZCBsb25nAGZsb2F0AGRvdWJsZQBzdGQ6OnN0cmluZwBzdGQ6OmJhc2ljX3N0cmluZzx1bnNpZ25lZCBjaGFyPgBzdGQ6OndzdHJpbmcAc3RkOjp1MTZzdHJpbmcAc3RkOjp1MzJzdHJpbmcAZW1zY3JpcHRlbjo6dmFsAGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PGNoYXI+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PHNpZ25lZCBjaGFyPgBlbXNjcmlwdGVuOjptZW1vcnlfdmlldzx1bnNpZ25lZCBjaGFyPgBlbXNjcmlwdGVuOjptZW1vcnlfdmlldzxzaG9ydD4AZW1zY3JpcHRlbjo6bWVtb3J5X3ZpZXc8dW5zaWduZWQgc2hvcnQ+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PGludD4AZW1zY3JpcHRlbjo6bWVtb3J5X3ZpZXc8dW5zaWduZWQgaW50PgBlbXNjcmlwdGVuOjptZW1vcnlfdmlldzxsb25nPgBlbXNjcmlwdGVuOjptZW1vcnlfdmlldzx1bnNpZ25lZCBsb25nPgBlbXNjcmlwdGVuOjptZW1vcnlfdmlldzxpbnQ4X3Q+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PHVpbnQ4X3Q+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PGludDE2X3Q+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PHVpbnQxNl90PgBlbXNjcmlwdGVuOjptZW1vcnlfdmlldzxpbnQzMl90PgBlbXNjcmlwdGVuOjptZW1vcnlfdmlldzx1aW50MzJfdD4AZW1zY3JpcHRlbjo6bWVtb3J5X3ZpZXc8ZmxvYXQ+AGVtc2NyaXB0ZW46Om1lbW9yeV92aWV3PGRvdWJsZT4ATlN0M19fMjEyYmFzaWNfc3RyaW5nSWNOU18xMWNoYXJfdHJhaXRzSWNFRU5TXzlhbGxvY2F0b3JJY0VFRUUATlN0M19fMjIxX19iYXNpY19zdHJpbmdfY29tbW9uSUxiMUVFRQAAhBcAABUPAAAIGAAA1g4AAAAAAAABAAAAPA8AAAAAAABOU3QzX18yMTJiYXNpY19zdHJpbmdJaE5TXzExY2hhcl90cmFpdHNJaEVFTlNfOWFsbG9jYXRvckloRUVFRQAACBgAAFwPAAAAAAAAAQAAADwPAAAAAAAATlN0M19fMjEyYmFzaWNfc3RyaW5nSXdOU18xMWNoYXJfdHJhaXRzSXdFRU5TXzlhbGxvY2F0b3JJd0VFRUUAAAgYAAC0DwAAAAAAAAEAAAA8DwAAAAAAAE5TdDNfXzIxMmJhc2ljX3N0cmluZ0lEc05TXzExY2hhcl90cmFpdHNJRHNFRU5TXzlhbGxvY2F0b3JJRHNFRUVFAAAACBgAAAwQAAAAAAAAAQAAADwPAAAAAAAATlN0M19fMjEyYmFzaWNfc3RyaW5nSURpTlNfMTFj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";
      if (!Ka(L)) { var Ma = L; L = e.locateFile ? e.locateFile(Ma, w) : w + Ma; } function Na() { try { if (A) return new Uint8Array(A); var a = ka(L); if (a) return a; if (fa) return fa(L); throw "both async and sync fetching of the wasm failed"; } catch (b) { y(b); } } function Oa() { return A || !ca && !v || "function" !== typeof fetch ? Promise.resolve().then(Na) : fetch(L, { credentials: "same-origin" }).then(function (a) { if (!a.ok) throw "failed to load wasm binary file at '" + L + "'"; return a.arrayBuffer() }).catch(function () { return Na() }) }
      function Pa(a) { for (; 0 < a.length;) { var b = a.shift(); if ("function" == typeof b) b(e); else { var c = b.za; "number" === typeof c ? void 0 === b.sa ? C.get(c)() : C.get(c)(b.sa) : c(void 0 === b.sa ? null : b.sa); } } } function Ra(a) { this.aa = a - 16; this.Na = function (b) { G[this.aa + 8 >> 2] = b; }; this.Ka = function (b) { G[this.aa + 0 >> 2] = b; }; this.La = function () { G[this.aa + 4 >> 2] = 0; }; this.Ja = function () { H[this.aa + 12 >> 0] = 0; }; this.Ma = function () { H[this.aa + 13 >> 0] = 0; }; this.Da = function (b, c) { this.Na(b); this.Ka(c); this.La(); this.Ja(); this.Ma(); }; }
      function Sa() { return 0 < Sa.ua } function Ta(a) { switch (a) { case 1: return 0; case 2: return 1; case 4: return 2; case 8: return 3; default: throw new TypeError("Unknown type size: " + a); } } var Ua = void 0; function M(a) { for (var b = ""; E[a];)b += Ua[E[a++]]; return b } var N = {}, O = {}, Va = {}; function Wa(a) { if (void 0 === a) return "_unknown"; a = a.replace(/[^a-zA-Z0-9_]/g, "$"); var b = a.charCodeAt(0); return 48 <= b && 57 >= b ? "_" + a : a }
      function Xa(a, b) { a = Wa(a); return (new Function("body", "return function " + a + '() {\n    "use strict";    return body.apply(this, arguments);\n};\n'))(b) } function Ya(a) { var b = Error, c = Xa(a, function (d) { this.name = a; this.message = d; d = Error(d).stack; void 0 !== d && (this.stack = this.toString() + "\n" + d.replace(/^Error(:[^\n]*)?\n/, "")); }); c.prototype = Object.create(b.prototype); c.prototype.constructor = c; c.prototype.toString = function () { return void 0 === this.message ? this.name : this.name + ": " + this.message }; return c }
      var P = void 0; function Q(a) { throw new P(a); } var Za = void 0; function $a(a) { throw new Za(a); } function R(a, b, c) { function d(h) { h = c(h); h.length !== a.length && $a("Mismatched type converter count"); for (var n = 0; n < a.length; ++n)S(a[n], h[n]); } a.forEach(function (h) { Va[h] = b; }); var f = Array(b.length), g = [], k = 0; b.forEach(function (h, n) { O.hasOwnProperty(h) ? f[n] = O[h] : (g.push(h), N.hasOwnProperty(h) || (N[h] = []), N[h].push(function () { f[n] = O[h]; ++k; k === g.length && d(f); })); }); 0 === g.length && d(f); }
      function S(a, b, c) { c = c || {}; if (!("argPackAdvance" in b)) throw new TypeError("registerType registeredInstance requires argPackAdvance"); var d = b.name; a || Q('type "' + d + '" must have a positive integer typeid pointer'); if (O.hasOwnProperty(a)) { if (c.Ca) return; Q("Cannot register type '" + d + "' twice"); } O[a] = b; delete Va[a]; N.hasOwnProperty(a) && (b = N[a], delete N[a], b.forEach(function (f) { f(); })); } function ab(a) { return { count: a.count, la: a.la, na: a.na, aa: a.aa, da: a.da, fa: a.fa, ga: a.ga } }
      function bb(a) { Q(a.$.da.ba.name + " instance already deleted"); } var cb = !1; function db() { } function eb(a) { --a.count.value; 0 === a.count.value && (a.fa ? a.ga.ka(a.fa) : a.da.ba.ka(a.aa)); }
      function fb(a) { if ("undefined" === typeof FinalizationGroup) return fb = function (b) { return b }, a; cb = new FinalizationGroup(function (b) { for (var c = b.next(); !c.done; c = b.next())c = c.value, c.aa ? eb(c) : console.warn("object already deleted: " + c.aa); }); fb = function (b) { cb.register(b, b.$, b.$); return b }; db = function (b) { cb.unregister(b.$); }; return fb(a) } var gb = void 0, hb = []; function ib() { for (; hb.length;) { var a = hb.pop(); a.$.la = !1; a["delete"](); } } function T() { } var jb = {};
      function kb(a, b, c) { if (void 0 === a[b].ea) { var d = a[b]; a[b] = function () { a[b].ea.hasOwnProperty(arguments.length) || Q("Function '" + c + "' called with an invalid number of arguments (" + arguments.length + ") - expects one of (" + a[b].ea + ")!"); return a[b].ea[arguments.length].apply(this, arguments) }; a[b].ea = []; a[b].ea[d.pa] = d; } }
      function lb(a, b, c) { e.hasOwnProperty(a) ? ((void 0 === c || void 0 !== e[a].ea && void 0 !== e[a].ea[c]) && Q("Cannot register public name '" + a + "' twice"), kb(e, a, a), e.hasOwnProperty(c) && Q("Cannot register multiple overloads of a function with the same number of arguments (" + c + ")!"), e[a].ea[c] = b) : (e[a] = b, void 0 !== c && (e[a].Qa = c)); } function mb(a, b, c, d, f, g, k, h) { this.name = a; this.constructor = b; this.ma = c; this.ka = d; this.ha = f; this.Aa = g; this.oa = k; this.ya = h; this.Fa = []; }
      function nb(a, b, c) { for (; b !== c;)b.oa || Q("Expected null or instance of " + c.name + ", got an instance of " + b.name), a = b.oa(a), b = b.ha; return a } function ob(a, b) { if (null === b) return this.ta && Q("null is not a valid " + this.name), 0; b.$ || Q('Cannot pass "' + V(b) + '" as a ' + this.name); b.$.aa || Q("Cannot pass deleted object as a pointer of type " + this.name); return nb(b.$.aa, b.$.da.ba, this.ba) }
      function pb(a, b) {
        if (null === b) { this.ta && Q("null is not a valid " + this.name); if (this.ra) { var c = this.Ga(); null !== a && a.push(this.ka, c); return c } return 0 } b.$ || Q('Cannot pass "' + V(b) + '" as a ' + this.name); b.$.aa || Q("Cannot pass deleted object as a pointer of type " + this.name); !this.qa && b.$.da.qa && Q("Cannot convert argument of type " + (b.$.ga ? b.$.ga.name : b.$.da.name) + " to parameter type " + this.name); c = nb(b.$.aa, b.$.da.ba, this.ba); if (this.ra) switch (void 0 === b.$.fa && Q("Passing raw pointer to smart pointer is illegal"),
          this.Oa) { case 0: b.$.ga === this ? c = b.$.fa : Q("Cannot convert argument of type " + (b.$.ga ? b.$.ga.name : b.$.da.name) + " to parameter type " + this.name); break; case 1: c = b.$.fa; break; case 2: if (b.$.ga === this) c = b.$.fa; else { var d = b.clone(); c = this.Ha(c, qb(function () { d["delete"](); })); null !== a && a.push(this.ka, c); } break; default: Q("Unsupporting sharing policy"); }return c
      }
      function rb(a, b) { if (null === b) return this.ta && Q("null is not a valid " + this.name), 0; b.$ || Q('Cannot pass "' + V(b) + '" as a ' + this.name); b.$.aa || Q("Cannot pass deleted object as a pointer of type " + this.name); b.$.da.qa && Q("Cannot convert argument of type " + b.$.da.name + " to parameter type " + this.name); return nb(b.$.aa, b.$.da.ba, this.ba) } function ub(a) { return this.fromWireType(I[a >> 2]) } function vb(a, b, c) { if (b === c) return a; if (void 0 === c.ha) return null; a = vb(a, b, c.ha); return null === a ? null : c.ya(a) } var wb = {};
      function xb(a, b) { for (void 0 === b && Q("ptr should not be undefined"); a.ha;)b = a.oa(b), a = a.ha; return wb[b] } function yb(a, b) { b.da && b.aa || $a("makeClassHandle requires ptr and ptrType"); !!b.ga !== !!b.fa && $a("Both smartPtrType and smartPtr must be specified"); b.count = { value: 1 }; return fb(Object.create(a, { $: { value: b } })) }
      function W(a, b, c, d) { this.name = a; this.ba = b; this.ta = c; this.qa = d; this.ra = !1; this.ka = this.Ha = this.Ga = this.wa = this.Oa = this.Ea = void 0; void 0 !== b.ha ? this.toWireType = pb : (this.toWireType = d ? ob : rb, this.ia = null); } function zb(a, b, c) { e.hasOwnProperty(a) || $a("Replacing nonexistant public symbol"); void 0 !== e[a].ea && void 0 !== c ? e[a].ea[c] = b : (e[a] = b, e[a].pa = c); }
      function Ab(a, b) { assert(0 <= a.indexOf("j"), "getDynCaller should only be called with i64 sigs"); var c = []; return function () { c.length = arguments.length; for (var d = 0; d < arguments.length; d++)c[d] = arguments[d]; var f; -1 != a.indexOf("j") ? f = c && c.length ? e["dynCall_" + a].apply(null, [b].concat(c)) : e["dynCall_" + a].call(null, b) : f = C.get(b).apply(null, c); return f } } function X(a, b) { a = M(a); var c = -1 != a.indexOf("j") ? Ab(a, b) : C.get(b); "function" !== typeof c && Q("unknown function pointer with signature " + a + ": " + b); return c }
      var Bb = void 0; function Cb(a) { a = Db(a); var b = M(a); Y(a); return b } function Eb(a, b) { function c(g) { f[g] || O[g] || (Va[g] ? Va[g].forEach(c) : (d.push(g), f[g] = !0)); } var d = [], f = {}; b.forEach(c); throw new Bb(a + ": " + d.map(Cb).join([", "])); } function Fb(a, b) { for (var c = [], d = 0; d < a; d++)c.push(G[(b >> 2) + d]); return c } function Gb(a) { for (; a.length;) { var b = a.pop(); a.pop()(b); } }
      function Hb(a) { var b = Function; if (!(b instanceof Function)) throw new TypeError("new_ called with constructor type " + typeof b + " which is not a function"); var c = Xa(b.name || "unknownFunctionName", function () { }); c.prototype = b.prototype; c = new c; a = b.apply(c, a); return a instanceof Object ? a : c }
      function Ib(a, b, c, d, f) {
        var g = b.length; 2 > g && Q("argTypes array size mismatch! Must at least get return value and 'this' types!"); var k = null !== b[1] && null !== c, h = !1; for (c = 1; c < b.length; ++c)if (null !== b[c] && void 0 === b[c].ia) { h = !0; break } var n = "void" !== b[0].name, l = "", m = ""; for (c = 0; c < g - 2; ++c)l += (0 !== c ? ", " : "") + "arg" + c, m += (0 !== c ? ", " : "") + "arg" + c + "Wired"; a = "return function " + Wa(a) + "(" + l + ") {\nif (arguments.length !== " + (g - 2) + ") {\nthrowBindingError('function " + a + " called with ' + arguments.length + ' arguments, expected " +
          (g - 2) + " args!');\n}\n"; h && (a += "var destructors = [];\n"); var q = h ? "destructors" : "null"; l = "throwBindingError invoker fn runDestructors retType classParam".split(" "); d = [Q, d, f, Gb, b[0], b[1]]; k && (a += "var thisWired = classParam.toWireType(" + q + ", this);\n"); for (c = 0; c < g - 2; ++c)a += "var arg" + c + "Wired = argType" + c + ".toWireType(" + q + ", arg" + c + "); // " + b[c + 2].name + "\n", l.push("argType" + c), d.push(b[c + 2]); k && (m = "thisWired" + (0 < m.length ? ", " : "") + m); a += (n ? "var rv = " : "") + "invoker(fn" + (0 < m.length ? ", " : "") + m + ");\n"; if (h) a +=
            "runDestructors(destructors);\n"; else for (c = k ? 1 : 2; c < b.length; ++c)g = 1 === c ? "thisWired" : "arg" + (c - 2) + "Wired", null !== b[c].ia && (a += g + "_dtor(" + g + "); // " + b[c].name + "\n", l.push(g + "_dtor"), d.push(b[c].ia)); n && (a += "var ret = retType.fromWireType(rv);\nreturn ret;\n"); l.push(a + "}\n"); return Hb(l).apply(null, d)
      } var Jb = [], Z = [{}, { value: void 0 }, { value: null }, { value: !0 }, { value: !1 }];
      function qb(a) { switch (a) { case void 0: return 1; case null: return 2; case !0: return 3; case !1: return 4; default: var b = Jb.length ? Jb.pop() : Z.length; Z[b] = { Ia: 1, value: a }; return b } } function V(a) { if (null === a) return "null"; var b = typeof a; return "object" === b || "array" === b || "function" === b ? a.toString() : "" + a } function Kb(a, b) { switch (b) { case 2: return function (c) { return this.fromWireType(Aa[c >> 2]) }; case 3: return function (c) { return this.fromWireType(Ba[c >> 3]) }; default: throw new TypeError("Unknown float type: " + a); } }
      function Lb(a, b, c) { switch (b) { case 0: return c ? function (d) { return H[d] } : function (d) { return E[d] }; case 1: return c ? function (d) { return F[d >> 1] } : function (d) { return sa[d >> 1] }; case 2: return c ? function (d) { return G[d >> 2] } : function (d) { return I[d >> 2] }; default: throw new TypeError("Unknown integer type: " + a); } } for (var Mb = [null, [], []], Nb = Array(256), Ob = 0; 256 > Ob; ++Ob)Nb[Ob] = String.fromCharCode(Ob); Ua = Nb; P = e.BindingError = Ya("BindingError"); Za = e.InternalError = Ya("InternalError");
      T.prototype.isAliasOf = function (a) { if (!(this instanceof T && a instanceof T)) return !1; var b = this.$.da.ba, c = this.$.aa, d = a.$.da.ba; for (a = a.$.aa; b.ha;)c = b.oa(c), b = b.ha; for (; d.ha;)a = d.oa(a), d = d.ha; return b === d && c === a }; T.prototype.clone = function () { this.$.aa || bb(this); if (this.$.na) return this.$.count.value += 1, this; var a = fb(Object.create(Object.getPrototypeOf(this), { $: { value: ab(this.$) } })); a.$.count.value += 1; a.$.la = !1; return a };
      T.prototype["delete"] = function () { this.$.aa || bb(this); this.$.la && !this.$.na && Q("Object already scheduled for deletion"); db(this); eb(this.$); this.$.na || (this.$.fa = void 0, this.$.aa = void 0); }; T.prototype.isDeleted = function () { return !this.$.aa }; T.prototype.deleteLater = function () { this.$.aa || bb(this); this.$.la && !this.$.na && Q("Object already scheduled for deletion"); hb.push(this); 1 === hb.length && gb && gb(ib); this.$.la = !0; return this }; W.prototype.Ba = function (a) { this.wa && (a = this.wa(a)); return a };
      W.prototype.va = function (a) { this.ka && this.ka(a); }; W.prototype.argPackAdvance = 8; W.prototype.readValueFromPointer = ub; W.prototype.deleteObject = function (a) { if (null !== a) a["delete"](); };
      W.prototype.fromWireType = function (a) {
        function b() { return this.ra ? yb(this.ba.ma, { da: this.Ea, aa: c, ga: this, fa: a }) : yb(this.ba.ma, { da: this, aa: a }) } var c = this.Ba(a); if (!c) return this.va(a), null; var d = xb(this.ba, c); if (void 0 !== d) { if (0 === d.$.count.value) return d.$.aa = c, d.$.fa = a, d.clone(); d = d.clone(); this.va(a); return d } d = this.ba.Aa(c); d = jb[d]; if (!d) return b.call(this); d = this.qa ? d.xa : d.pointerType; var f = vb(c, this.ba, d.ba); return null === f ? b.call(this) : this.ra ? yb(d.ba.ma, { da: d, aa: f, ga: this, fa: a }) : yb(d.ba.ma,
          { da: d, aa: f })
      }; e.getInheritedInstanceCount = function () { return Object.keys(wb).length }; e.getLiveInheritedInstances = function () { var a = [], b; for (b in wb) wb.hasOwnProperty(b) && a.push(wb[b]); return a }; e.flushPendingDeletes = ib; e.setDelayFunction = function (a) { gb = a; hb.length && gb && gb(ib); }; Bb = e.UnboundTypeError = Ya("UnboundTypeError"); e.count_emval_handles = function () { for (var a = 0, b = 5; b < Z.length; ++b)void 0 !== Z[b] && ++a; return a }; e.get_first_emval = function () { for (var a = 5; a < Z.length; ++a)if (void 0 !== Z[a]) return Z[a]; return null };
      var Pb = "function" === typeof atob ? atob : function (a) {
        var b = "", c = 0; a = a.replace(/[^A-Za-z0-9\+\/=]/g, ""); do {
          var d = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(a.charAt(c++)); var f = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(a.charAt(c++)); var g = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(a.charAt(c++)); var k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(a.charAt(c++)); d = d <<
            2 | f >> 4; f = (f & 15) << 4 | g >> 2; var h = (g & 3) << 6 | k; b += String.fromCharCode(d); 64 !== g && (b += String.fromCharCode(f)); 64 !== k && (b += String.fromCharCode(h));
        } while (c < a.length); return b
      };
      function ka(a) { if (Ka(a)) { a = a.slice(La.length); if ("boolean" === typeof da && da) { try { var b = Buffer.from(a, "base64"); } catch (g) { b = new Buffer(a, "base64"); } var c = new Uint8Array(b.buffer, b.byteOffset, b.byteLength); } else try { var d = Pb(a), f = new Uint8Array(d.length); for (b = 0; b < d.length; ++b)f[b] = d.charCodeAt(b); c = f; } catch (g) { throw Error("Converting base64 string to bytes failed."); } return c } } Ea.push({ za: function () { Qb(); } });
      var Sb = {
        e: function (a, b, c, d) { y("Assertion failed: " + (a ? D(E, a, void 0) : "") + ", at: " + [b ? b ? D(E, b, void 0) : "" : "unknown filename", c, d ? d ? D(E, d, void 0) : "" : "unknown function"]); }, p: function (a) { return Rb(a + 16) + 16 }, o: function (a, b, c) { (new Ra(a)).Da(b, c); "uncaught_exception" in Sa ? Sa.ua++ : Sa.ua = 1; throw a; }, b: C, m: function (a, b, c, d, f) {
          var g = Ta(c); b = M(b); S(a, {
            name: b, fromWireType: function (k) { return !!k }, toWireType: function (k, h) { return h ? d : f }, argPackAdvance: 8, readValueFromPointer: function (k) {
              if (1 === c) var h = H; else if (2 ===
                c) h = F; else if (4 === c) h = G; else throw new TypeError("Unknown boolean type size: " + b); return this.fromWireType(h[k >> g])
            }, ia: null
          });
        }, w: function (a, b, c, d, f, g, k, h, n, l, m, q, r) {
          m = M(m); g = X(f, g); h && (h = X(k, h)); l && (l = X(n, l)); r = X(q, r); var x = Wa(m); lb(x, function () { Eb("Cannot construct " + m + " due to unbound types", [d]); }); R([a, b, c], d ? [d] : [], function (t) {
            t = t[0]; if (d) { var va = t.ba; var ha = va.ma; } else ha = T.prototype; t = Xa(x, function () {
              if (Object.getPrototypeOf(this) !== Qa) throw new P("Use 'new' to construct " + m); if (void 0 === U.ja) throw new P(m +
                " has no accessible constructor"); var sb = U.ja[arguments.length]; if (void 0 === sb) throw new P("Tried to invoke ctor of " + m + " with invalid number of parameters (" + arguments.length + ") - expected (" + Object.keys(U.ja).toString() + ") parameters instead!"); return sb.apply(this, arguments)
            }); var Qa = Object.create(ha, { constructor: { value: t } }); t.prototype = Qa; var U = new mb(m, t, Qa, r, va, g, h, l); va = new W(m, U, !0, !1); ha = new W(m + "*", U, !1, !1); var tb = new W(m + " const*", U, !1, !0); jb[a] = { pointerType: ha, xa: tb }; zb(x, t); return [va,
              ha, tb]
          });
        }, r: function (a, b, c, d, f, g) {
          assert(0 < b); var k = Fb(b, c); f = X(d, f); var h = [g], n = []; R([], [a], function (l) {
            l = l[0]; var m = "constructor " + l.name; void 0 === l.ba.ja && (l.ba.ja = []); if (void 0 !== l.ba.ja[b - 1]) throw new P("Cannot register multiple constructors with identical number of parameters (" + (b - 1) + ") for class '" + l.name + "'! Overload resolution is currently only performed using the parameter count, not actual type info!"); l.ba.ja[b - 1] = function () { Eb("Cannot construct " + l.name + " due to unbound types", k); }; R([],
              k, function (q) { l.ba.ja[b - 1] = function () { arguments.length !== b - 1 && Q(m + " called with " + arguments.length + " arguments, expected " + (b - 1)); n.length = 0; h.length = b; for (var r = 1; r < b; ++r)h[r] = q[r].toWireType(n, arguments[r - 1]); r = f.apply(null, h); Gb(n); return q[0].fromWireType(r) }; return [] }); return []
          });
        }, f: function (a, b, c, d, f, g, k, h) {
          var n = Fb(c, d); b = M(b); g = X(f, g); R([], [a], function (l) {
            function m() { Eb("Cannot call " + q + " due to unbound types", n); } l = l[0]; var q = l.name + "." + b; h && l.ba.Fa.push(b); var r = l.ba.ma, x = r[b]; void 0 ===
              x || void 0 === x.ea && x.className !== l.name && x.pa === c - 2 ? (m.pa = c - 2, m.className = l.name, r[b] = m) : (kb(r, b, q), r[b].ea[c - 2] = m); R([], n, function (t) { t = Ib(q, t, l, g, k); void 0 === r[b].ea ? (t.pa = c - 2, r[b] = t) : r[b].ea[c - 2] = t; return [] }); return []
          });
        }, v: function (a, b) { b = M(b); S(a, { name: b, fromWireType: function (c) { var d = Z[c].value; 4 < c && 0 === --Z[c].Ia && (Z[c] = void 0, Jb.push(c)); return d }, toWireType: function (c, d) { return qb(d) }, argPackAdvance: 8, readValueFromPointer: ub, ia: null }); }, l: function (a, b, c) {
          c = Ta(c); b = M(b); S(a, {
            name: b, fromWireType: function (d) { return d },
            toWireType: function (d, f) { if ("number" !== typeof f && "boolean" !== typeof f) throw new TypeError('Cannot convert "' + V(f) + '" to ' + this.name); return f }, argPackAdvance: 8, readValueFromPointer: Kb(b, c), ia: null
          });
        }, x: function (a, b, c, d, f, g) { var k = Fb(b, c); a = M(a); f = X(d, f); lb(a, function () { Eb("Cannot call " + a + " due to unbound types", k); }, b - 1); R([], k, function (h) { h = [h[0], null].concat(h.slice(1)); zb(a, Ib(a, h, null, f, g), b - 1); return [] }); }, d: function (a, b, c, d, f) {
          function g(l) { return l } b = M(b); -1 === f && (f = 4294967295); var k = Ta(c);
          if (0 === d) { var h = 32 - 8 * c; g = function (l) { return l << h >>> h }; } var n = -1 != b.indexOf("unsigned"); S(a, { name: b, fromWireType: g, toWireType: function (l, m) { if ("number" !== typeof m && "boolean" !== typeof m) throw new TypeError('Cannot convert "' + V(m) + '" to ' + this.name); if (m < d || m > f) throw new TypeError('Passing a number "' + V(m) + '" from JS side to C/C++ side to an argument of type "' + b + '", which is outside the valid range [' + d + ", " + f + "]!"); return n ? m >>> 0 : m | 0 }, argPackAdvance: 8, readValueFromPointer: Lb(b, k, 0 !== d), ia: null });
        }, c: function (a,
          b, c) { function d(g) { g >>= 2; return new f(za, I[g + 1], I[g]) } var f = [Int8Array, Uint8Array, Int16Array, Uint16Array, Int32Array, Uint32Array, Float32Array, Float64Array][b]; c = M(c); S(a, { name: c, fromWireType: d, argPackAdvance: 8, readValueFromPointer: d }, { Ca: !0 }); }, i: function (a, b) {
            b = M(b); var c = "std::string" === b; S(a, {
              name: b, fromWireType: function (d) {
                var f = I[d >> 2]; if (c) for (var g = d + 4, k = 0; k <= f; ++k) { var h = d + 4 + k; if (k == f || 0 == E[h]) { g = g ? D(E, g, h - g) : ""; if (void 0 === n) var n = g; else n += String.fromCharCode(0), n += g; g = h + 1; } } else {
                  n = Array(f);
                  for (k = 0; k < f; ++k)n[k] = String.fromCharCode(E[d + 4 + k]); n = n.join("");
                } Y(d); return n
              }, toWireType: function (d, f) {
                f instanceof ArrayBuffer && (f = new Uint8Array(f)); var g = "string" === typeof f; g || f instanceof Uint8Array || f instanceof Uint8ClampedArray || f instanceof Int8Array || Q("Cannot pass non-string to std::string"); var k = (c && g ? function () { for (var l = 0, m = 0; m < f.length; ++m) { var q = f.charCodeAt(m); 55296 <= q && 57343 >= q && (q = 65536 + ((q & 1023) << 10) | f.charCodeAt(++m) & 1023); 127 >= q ? ++l : l = 2047 >= q ? l + 2 : 65535 >= q ? l + 3 : l + 4; } return l } :
                  function () { return f.length })(), h = Rb(4 + k + 1); I[h >> 2] = k; if (c && g) pa(f, h + 4, k + 1); else if (g) for (g = 0; g < k; ++g) { var n = f.charCodeAt(g); 255 < n && (Y(h), Q("String has UTF-16 code units that do not fit in 8 bits")); E[h + 4 + g] = n; } else for (g = 0; g < k; ++g)E[h + 4 + g] = f[g]; null !== d && d.push(Y, h); return h
              }, argPackAdvance: 8, readValueFromPointer: ub, ia: function (d) { Y(d); }
            });
          }, h: function (a, b, c) {
            c = M(c); if (2 === b) { var d = ra; var f = ta; var g = ua; var k = function () { return sa }; var h = 1; } else 4 === b && (d = wa, f = xa, g = ya, k = function () { return I }, h = 2); S(a, {
              name: c,
              fromWireType: function (n) { for (var l = I[n >> 2], m = k(), q, r = n + 4, x = 0; x <= l; ++x) { var t = n + 4 + x * b; if (x == l || 0 == m[t >> h]) r = d(r, t - r), void 0 === q ? q = r : (q += String.fromCharCode(0), q += r), r = t + b; } Y(n); return q }, toWireType: function (n, l) { "string" !== typeof l && Q("Cannot pass non-string to C++ string type " + c); var m = g(l), q = Rb(4 + m + b); I[q >> 2] = m >> h; f(l, q + 4, m + b); null !== n && n.push(Y, q); return q }, argPackAdvance: 8, readValueFromPointer: ub, ia: function (n) { Y(n); }
            });
          }, n: function (a, b) {
            b = M(b); S(a, {
              Pa: !0, name: b, argPackAdvance: 0, fromWireType: function () { },
              toWireType: function () { }
            });
          }, g: function () { y(); }, s: function (a, b, c) { E.copyWithin(a, b, b + c); }, t: function () { y("OOM"); }, u: function () { return 0 }, q: function () { }, k: function (a, b, c, d) { for (var f = 0, g = 0; g < c; g++) { for (var k = G[b + 8 * g >> 2], h = G[b + (8 * g + 4) >> 2], n = 0; n < h; n++) { var l = E[k + n], m = Mb[a]; 0 === l || 10 === l ? ((1 === a ? ma : z)(D(m, 0)), m.length = 0) : m.push(l); } f += h; } G[d >> 2] = f; return 0 }, a: B, j: function () { }
      };
      (function () {
        function a(f) { e.asm = f.exports; K--; e.monitorRunDependencies && e.monitorRunDependencies(K); 0 == K && (Ja && (f = Ja, Ja = null, f())); } function b(f) { a(f.instance); } function c(f) { return Oa().then(function (g) { return WebAssembly.instantiate(g, d) }).then(f, function (g) { z("failed to asynchronously prepare wasm: " + g); y(g); }) } var d = { a: Sb }; K++; e.monitorRunDependencies && e.monitorRunDependencies(K); if (e.instantiateWasm) try { return e.instantiateWasm(d, a) } catch (f) {
          return z("Module.instantiateWasm callback failed with error: " +
            f), !1
        } (function () { if (A || "function" !== typeof WebAssembly.instantiateStreaming || Ka(L) || "function" !== typeof fetch) return c(b); fetch(L, { credentials: "same-origin" }).then(function (f) { return WebAssembly.instantiateStreaming(f, d).then(b, function (g) { z("wasm streaming compile failed: " + g); z("falling back to ArrayBuffer instantiation"); return c(b) }) }); })(); return {}
      })(); var Qb = e.___wasm_call_ctors = function () { return (Qb = e.___wasm_call_ctors = e.asm.y).apply(null, arguments) };
      e._siphash = function () { return (e._siphash = e.asm.z).apply(null, arguments) }; e._DecodeBase58 = function () { return (e._DecodeBase58 = e.asm.A).apply(null, arguments) }; e._EncodeBase58 = function () { return (e._EncodeBase58 = e.asm.B).apply(null, arguments) }; e.__ripemd160 = function () { return (e.__ripemd160 = e.asm.C).apply(null, arguments) }; e._hmac_sha512_oneline = function () { return (e._hmac_sha512_oneline = e.asm.D).apply(null, arguments) }; e._pbkdf2_hmac_sha512 = function () { return (e._pbkdf2_hmac_sha512 = e.asm.E).apply(null, arguments) };
      var Rb = e._malloc = function () { return (Rb = e._malloc = e.asm.F).apply(null, arguments) }, Y = e._free = function () { return (Y = e._free = e.asm.G).apply(null, arguments) }; e._md5sum = function () { return (e._md5sum = e.asm.H).apply(null, arguments) }; e._sha3 = function () { return (e._sha3 = e.asm.I).apply(null, arguments) }; e._secp256k1_context_create = function () { return (e._secp256k1_context_create = e.asm.J).apply(null, arguments) }; e._secp256k1_ec_pubkey_parse = function () { return (e._secp256k1_ec_pubkey_parse = e.asm.K).apply(null, arguments) };
      e._secp256k1_ec_pubkey_serialize = function () { return (e._secp256k1_ec_pubkey_serialize = e.asm.L).apply(null, arguments) }; e._secp256k1_ecdsa_signature_parse_der = function () { return (e._secp256k1_ecdsa_signature_parse_der = e.asm.M).apply(null, arguments) }; e._secp256k1_ecdsa_signature_serialize_der = function () { return (e._secp256k1_ecdsa_signature_serialize_der = e.asm.N).apply(null, arguments) };
      e._secp256k1_ecdsa_signature_serialize_compact = function () { return (e._secp256k1_ecdsa_signature_serialize_compact = e.asm.O).apply(null, arguments) }; e._secp256k1_ecdsa_verify = function () { return (e._secp256k1_ecdsa_verify = e.asm.P).apply(null, arguments) }; e._secp256k1_ec_pubkey_create = function () { return (e._secp256k1_ec_pubkey_create = e.asm.Q).apply(null, arguments) }; e._secp256k1_ec_pubkey_tweak_add = function () { return (e._secp256k1_ec_pubkey_tweak_add = e.asm.R).apply(null, arguments) };
      e._secp256k1_context_randomize = function () { return (e._secp256k1_context_randomize = e.asm.S).apply(null, arguments) }; e._secp256k1_ecdsa_recoverable_signature_parse_compact = function () { return (e._secp256k1_ecdsa_recoverable_signature_parse_compact = e.asm.T).apply(null, arguments) }; e._secp256k1_ecdsa_recoverable_signature_serialize_compact = function () { return (e._secp256k1_ecdsa_recoverable_signature_serialize_compact = e.asm.U).apply(null, arguments) };
      e._secp256k1_ecdsa_sign_recoverable = function () { return (e._secp256k1_ecdsa_sign_recoverable = e.asm.V).apply(null, arguments) }; e._secp256k1_ecdsa_recover = function () { return (e._secp256k1_ecdsa_recover = e.asm.W).apply(null, arguments) }; var Db = e.___getTypeName = function () { return (Db = e.___getTypeName = e.asm.X).apply(null, arguments) }; e.___embind_register_native_and_builtin_types = function () { return (e.___embind_register_native_and_builtin_types = e.asm.Y).apply(null, arguments) };
      e.dynCall_jiji = function () { return (e.dynCall_jiji = e.asm.Z).apply(null, arguments) }; e.getValue = function (a, b) { b = b || "i8"; "*" === b.charAt(b.length - 1) && (b = "i32"); switch (b) { case "i1": return H[a >> 0]; case "i8": return H[a >> 0]; case "i16": return F[a >> 1]; case "i32": return G[a >> 2]; case "i64": return G[a >> 2]; case "float": return Aa[a >> 2]; case "double": return Ba[a >> 3]; default: y("invalid type for getValue: " + b); }return null }; var Tb; Ja = function Ub() { Tb || Vb(); Tb || (Ja = Ub); };
      function Vb() { function a() { if (!Tb && (Tb = !0, e.calledRun = !0, !na)) { Pa(Ea); Pa(Fa); aa(e); if (e.onRuntimeInitialized) e.onRuntimeInitialized(); if (e.postRun) for ("function" == typeof e.postRun && (e.postRun = [e.postRun]); e.postRun.length;) { var b = e.postRun.shift(); Ga.unshift(b); } Pa(Ga); } } if (!(0 < K)) { if (e.preRun) for ("function" == typeof e.preRun && (e.preRun = [e.preRun]); e.preRun.length;)Ha(); Pa(Da); 0 < K || (e.setStatus ? (e.setStatus("Running..."), setTimeout(function () { setTimeout(function () { e.setStatus(""); }, 1); a(); }, 1)) : a()); } }
      e.run = Vb; if (e.preInit) for ("function" == typeof e.preInit && (e.preInit = [e.preInit]); 0 < e.preInit.length;)e.preInit.pop()(); Vb();


      return Module.ready
    }
  );
})();

// else if (typeof define === 'function' && define['amd'])
// define([], function () { return Module; });
// else if (typeof exports === 'object')
// exports["Module"] = Module;

function Address(window) {
  let Buffer = window.Buffer;
  let defArgs = window.defArgs;
  let getBuffer = window.getBuffer;
  let BF = Buffer.from;
  let BC = Buffer.concat;
  let O = window.OPCODE;

  class PrivateKey {
    /**
    * The class for creating private key object.
    *
    * :parameters:
    *    :k: (optional) private key in HEX,  bytes string or WIF format. In case no key specified new random private key will be created
    * :param compressed: (optional) if set to ``true`` private key corresponding compressed public key, by default is ``true``. Recommended use only compressed public key.
    * :param testnet: (optional) flag for testnet network, by default is ``false``.
    */
    constructor(k, A = {}) {
      defArgs(A, { compressed: null, testnet: false });
      if (k === undefined) {
        if (A.compressed === null) A.compressed = true;
        /**
        * flag for compressed type of corresponding public key (boolean)
        */
        this.compressed = A.compressed;
        /**
         * flag for testnet network (boolean)
         */
        this.testnet = A.testnet;
        this.key = window.createPrivateKey({ wif: false });
        /**
         * private key in HEX (string)
         */
        this.hex = this.key.hex();
        /**
         * private key in WIF format (string)
         */
        this.wif = window.privateKeyToWif(this.key, A);
      } else {
        if (window.isString(k)) {
          if (window.isHex(k)) {
            if (A.compressed === null) A.compressed = true;
            this.key = BF(k, 'hex');
            this.compressed = A.compressed;
            this.testnet = A.testnet;
            this.hex = this.key.hex();
            this.wif = window.privateKeyToWif(this.key, A);
          } else {
            this.wif = k;
            this.key = window.wifToPrivateKey(k, { hex: false });
            this.hex = this.key.hex();
            this.compressed = ![window.MAINNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX,
            window.TESTNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX].includes(k[0]);
            this.testnet = [window.TESTNET_PRIVATE_KEY_COMPRESSED_PREFIX,
            window.TESTNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX].includes(k[0]);

          }
        } else {
          k = BF(k);
          if (k.length !== 32) throw new Error('private key invalid');
          if (A.compressed === null) A.compressed = true;
          this.compressed = A.compressed;
          this.testnet = A.testnet;
          this.key = k;
          this.hex = this.key.hex();
          this.wif = window.privateKeyToWif(this.key, A);
        }
      }
    }
  }

  PrivateKey.prototype.toString = function () {
    return `${this.wif}`;
  };


  class PublicKey {
    /**
    * The class for creating public key object.
    *
    * :parameters:
    *   :k: one of this types allowed:
    *
    *           -- private key is instance of ``PrivateKey`` class
    *
    *           -- private key HEX encoded string
    *
    *           -- private key 32 bytes string
    *
    *           -- private key in WIF format
    *
    *           -- public key in HEX encoded string
    *
    *           -- public key [33/65] bytes string
    *
    * *In case no key specified with HEX or bytes string you have to provide flag for testnet
    * and compressed key. WIF format and* ``PrivateKey`` *instance already contain this flags.
    * For HEX or bytes public key only testnet flag has the meaning, comressed flag is determined
    * according to the length of key.*
    *
    * :param compressed: (optional) if set to ``true`` private key corresponding compressed public key, by default is ``true``. Recommended use only compressed public key.
    * :param testnet: (optional) flag for testnet network, by default is ``false``.
    */
    constructor(k, A = {}) {
      defArgs(A, { compressed: null, testnet: false });
      /**
       * flag for compressed type of corresponding public key (boolean)
       */
      this.compressed = A.compressed;
      /**
       * flag for testnet network (boolean)
       */
      this.testnet = A.testnet;
      if (k instanceof PrivateKey) {
        A.testnet = k.testnet;
        A.compressed = k.compressed;
        k = k.wif;
      }

      if (window.isString(k)) {
        if (window.isHex(k)) {
          k = BF(k, 'hex');
          if (A.compressed === null) A.compressed = true;
        }
        else if (window.isWifValid(k)) {
          this.compressed = ![window.MAINNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX,
          window.TESTNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX].includes(k[0]);
          this.testnet = [window.TESTNET_PRIVATE_KEY_COMPRESSED_PREFIX,
          window.TESTNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX].includes(k[0]);
          k = window.privateToPublicKey(k, { compressed: this.compressed, testnet: this.testnet, hex: false });
        } else throw new Error('private/public key invalid');
      } else k = BF(k);
      if (k.length === 32) {
        if (A.compressed === null) A.compressed = true;
        this.key = window.privateToPublicKey(k, { compressed: A.compressed, testnet: A.testnet, hex: false });
        this.compressed = A.compressed;
        this.testnet = A.testnet;
        this.hex = this.key.hex();
      } else if (window.isPublicKeyValid(k)) {
        /**
         * public key in HEX (string)
         */
        this.hex = k.hex();
        this.key = k;
        this.compressed = (this.key.length === 33);
        this.testnet = A.testnet;
      } else throw new Error('private/public key invalid');
    }
  }

  PublicKey.prototype.toString = function () {
    return `${this.hex}`;
  };


  class Address {
    /**
    * The class for Address object.
    *
    * :parameters:
    *   :k: (optional) one of this types allowed:
    *
    *           -- private key WIF format
    *
    *           -- instance of ``PrivateKey``
    *
    *           -- private key HEX encoded string
    *
    *           -- instance of ``PublicKey``
    *
    * :param addressType: (optional) P2PKH, PUBKEY, P2WPKH, P2SH_P2WPKH.
    * :param testnet: (optional) flag for testnet network, by default is ``false``.
    * :param compressed: (optional) if set to ``true`` private key corresponding compressed public key, by default is ``true``. Recommended use only compressed public key.
    *
    * *In case instance is created from WIF private key,* ``PrivateKey`` *or* ``PublicKey`` *compressed and testnet flags
    * already contain in initial key parameter and will be ignored.*
    */
    constructor(k, A = {}) {
      defArgs(A, { addressType: null, testnet: false, compressed: null });

      if (k === undefined) {
        if (A.compressed === null) A.compressed = true;
        /**
         * instance of ``PrivateKey`` class
         */
        this.privateKey = new PrivateKey(undefined, A);
        /**
         * instance of ``PublicKey`` class
         */
        this.publicKey = new PublicKey(this.privateKey, A);
      } else if (window.isString(k)) {
        if (window.isWifValid(k)) {
          this.privateKey = new PrivateKey(k, A);
          A.compressed = this.privateKey.compressed;
          this.publicKey = new PublicKey(this.privateKey, A);
          A.testnet = this.privateKey.testnet;
        }
        else if (window.isHex(k)) {
          if (A.compressed === null) A.compressed = true;
          k = BF(k, 'hex');
        }
        else {
          throw new Error('private/public key invalid');
        }
      }
      else if (k instanceof PrivateKey) {
        this.privateKey = k;
        A.testnet = k.testnet;
        A.compressed = k.compressed;
        this.publicKey = new PublicKey(this.privateKey, A);
      } else if (k instanceof PublicKey) {
        A.testnet = k.testnet;
        A.compressed = k.compressed;
        this.publicKey = k;
      } else {
        if (!Buffer.isBuffer(k)) k = BF(k);
      }

      if (Buffer.isBuffer(k)) {
        if (k.length === 32) {
          if (A.compressed === null) A.compressed = true;
          this.privateKey = new PrivateKey(k, A);
          this.publicKey = new PublicKey(this.privateKey, A);
        } else if (window.isPublicKeyValid(k)) {

          this.publicKey = new PublicKey(k, A);

          A.compressed = this.publicKey.compressed;
        } else throw new Error('private/public key invalid');
      }

      /**
       * flag for testnet network address  (boolean)
       */
      this.testnet = A.testnet;


      if (A.addressType === null) {
        if (A.compressed === false) A.addressType = "P2PKH";
        else A.addressType = "P2WPKH";
      }



      if (!["P2PKH", "PUBKEY", "P2WPKH", "P2SH_P2WPKH"].includes(A.addressType)) {
        throw new Error('address type invalid');
      }

      /**
       * address type (string)
       */
      this.type = A.addressType;
      if (this.type === 'PUBKEY') {
        this.publicKeyScript = BC([window.opPushData(this.publicKey.key), BF([O.OP_CHECKSIG])]);
        this.publicKeyScriptHex = this.publicKeyScript.hex();
      }
      this.witnessVersion = (this.type === "P2WPKH") ? 0 : null;
      if (this.type === "P2SH_P2WPKH") {
        /**
        * flag for script hash address (boolean)
        */
        this.scriptHash = true;
        /**
        * redeeem script, only for P2SH_P2WPKH (bytes)
        */
        this.redeemScript = window.publicKeyTo_P2SH_P2WPKH_Script(this.publicKey.key);
        /**
        * redeeem script HEX, only for P2SH_P2WPKH (string)
        */
        this.redeemScriptHex = this.redeemScript.hex();
        /**
        * address hash
        */
        this.hash = window.hash160(this.redeemScript);
        this.witnessVersion = null;
      } else {
        this.scriptHash = false;
        this.hash = window.hash160(this.publicKey.key);
      }
      /**
       * address hash HEX (string)
       */
      this.hashHex = this.hash.hex();
      this.testnet = A.testnet;
      /**
       * address in base58 or bech32 encoding (string)
       */
      this.address = window.hashToAddress(this.hash, {
        scriptHash: this.scriptHash,
        witnessVersion: this.witnessVersion, testnet: this.testnet
      });
    }
  }

  Address.prototype.toString = function () {
    return `${this.address}`;
  };

  class ScriptAddress {
    constructor(s, A = {}) {
      defArgs(A, { witnessVersion: 0, testnet: false });
      this.witnessVersion = A.witnessVersion;
      this.testnet = A.testnet;
      s = getBuffer(s);
      this.script = s;
      this.scriptHex = s.hex();
      if (this.witnessVersion === null) this.hash = window.hash160(this.script);
      else this.hash = window.sha256(this.script);
      this.scriptOpcodes = window.decodeScript(this.script);
      this.scriptOpcodesAsm = window.decodeScript(this.script, { asm: true });
      this.address = window.hashToAddress(this.hash, {
        scriptHash: true,
        witnessVersion: this.witnessVersion, testnet: this.testnet
      });
    }

    static multisig(n, m, keyList, A = {}) {
      if ((n > 15) || (m > 15) || (n > m) || (n < 1) || (m < 1))
        throw new Error('invalid n of m maximum 15 of 15 multisig allowed');
      if (keyList.length !== m)
        throw new Error('invalid address list count');
      let s = [BF([0x50 + n])];
      for (let k of keyList) {
        if (window.isString(k)) {
          if (window.isHex(k)) k = BF(k, 'hex');
          else if (window.isWifValid(k)) k = window.privateToPublicKey(k, { hex: false });
          else throw new Error('invalid key in key list');
        }
        if (k instanceof Address) k = k.publicKey.key;
        if (k instanceof PrivateKey) k = window.privateToPublicKey(k.publicKey.key);
        if (!Buffer.isBuffer(k)) k = BF(k);

        if (k.length === 32) k = window.privateToPublicKey(k);
        if (k.length !== 33) throw new Error('invalid public key list element size');
        s.push(BC([BF(window.intToVarInt(k.length)), k]));
      }
      s.push(BF([0x50 + m, O.OP_CHECKMULTISIG]));
      s = BC(s);
      return new ScriptAddress(s, A);
    }
  }

  ScriptAddress.prototype.toString = function () {
    return `${this.address}`;
  };

  window.PrivateKey = PrivateKey;
  window.PublicKey = PublicKey;
  window.ScriptAddress = ScriptAddress;
  window.Address = Address;
}

function Transation(window) {
  let s2rh = window.s2rh;
  let rh2s = window.rh2s;
  let Buffer = window.Buffer;
  window.BN;
  let isBuffer = window.Buffer.isBuffer;
  let ARGS = window.defArgs;
  let getBuffer = window.getBuffer;
  let BF = Buffer.from;
  let BA = Buffer.alloc;
  let BC = Buffer.concat;
  let O = window.OPCODE;
  let iS = window.isString;

  class Transaction {

    /**
    * The class for Transaction object
    *
    * :param rawTx: (optional) raw transaction in bytes or HEX encoded string, if no raw transaction provided well be created new empty transaction template.
    * :param format: (optional) "raw" or "decoded" format. Raw format is mean that all transaction represented in bytes for best performance. Decoded transaction is represented in human readable format using base68, hex, bech32, asm and opcodes. By default "decoded" format using.
    * :param version: (optional)  transaction version for new template, by default is 2.
    * :param lockTime: (optional)  transaction lock time for new template, by default is 0.
    * :param testnet: (optional) flag for testnet network, by default is ``false``.
    * :param autoCommit: (optional) boolean, by default is ``true``.
    * :param keepRawTx: (optional) boolean, by default is ``false``.
    */
    constructor(A = {}) {
      ARGS(A, {
        rawTx: null, format: 'decoded', version: 2,
        lockTime: 0, testnet: false, autoCommit: true, keepRawTx: false
      });
      if (!["decoded", "raw"].includes(A.format)) throw new Error('format error, raw or decoded allowed');
      this.autoCommit = A.autoCommit;
      this.format = A.format;
      this.testnet = A.testnet;
      this.segwit = false;
      this.txId = null;
      this.hash = null;
      this.version = A.version;
      this.size = 0;
      this.vSize = 0;
      this.bSize = 0;
      this.vIn = {};
      this.vOut = {};
      this.rawTx = null;
      this.blockHash = null;
      this.confirmations = null;
      this.time = null;
      this.blockTime = null;
      this.lockTime = A.lockTime;
      this.blockIndex = null;
      this.coinbase = false;
      this.fee = null;
      this.data = null;
      this.amount = null;
      if (A.rawTx === null) return;
      let tx = getBuffer(A.rawTx);
      this.amount = 0;
      let sw = 0, swLen = 0;
      let start = (tx.__offset === undefined) ? 0 : tx.__offset;
      this.version = tx.readInt(4);
      let n = tx.readVarInt();

      if (n[0] === 0) {
        // segwit format
        sw = 1;
        this.flag = tx.read(1);
        n = tx.readVarInt();
      }
      // inputs
      let ic = window.varIntToInt(n);
      for (let k = 0; k < ic; k++)
        this.vIn[k] = {
          txId: tx.read(32),
          vOut: tx.readInt(4),
          scriptSig: tx.read(window.varIntToInt(tx.readVarInt())),
          sequence: tx.readInt(4)
        };
      // outputs
      let oc = window.varIntToInt(tx.readVarInt());
      for (let k = 0; k < oc; k++) {
        this.vOut[k] = {};
        this.vOut[k].value = tx.readInt(8);
        this.amount += this.vOut[k].value;
        this.vOut[k].scriptPubKey = tx.read(window.varIntToInt(tx.readVarInt()));
        let s = window.parseScript(this.vOut[k].scriptPubKey);
        this.vOut[k].nType = s.nType;
        this.vOut[k].type = s.type;
        if ((this.data === null) && (s.type === 3)) this.data = s.data;
        if (s.addressHash !== undefined) {
          this.vOut[k].addressHash = s.addressHash;
          this.vOut[k].reqSigs = s.reqSigs;
        }
      }

      // witness
      if (sw) {
        sw = tx.__offset - start;
        for (let k = 0; k < ic; k++) {
          this.vIn[k].txInWitness = [];
          let t = window.varIntToInt(tx.readVarInt());
          for (let q = 0; q < t; q++)
            this.vIn[k].txInWitness.push(tx.read(window.varIntToInt(tx.readVarInt())));
        }
        swLen = (tx.__offset - start) - sw + 2;
      }
      this.lockTime = tx.readInt(4);
      let end = tx.__offset;
      this.rawTx = tx.slice(start, end);
      this.size = end - start;
      this.bSize = end - start - swLen;
      this.weight = this.bSize * 3 + this.size;
      this.vSize = Math.ceil(this.weight / 4);
      this.coinbase = !!((ic === 1) && (this.vIn[0].txId.equals(Buffer(32))) && (this.vIn[0].vOut === 0xffffffff));

      if (sw > 0) {
        this.segwit = true;
        this.hash = window.sha256(this.rawTx);
        this.txId = window.sha256(BC([this.rawTx.slice(0, 4),
        this.rawTx.slice(6, sw), this.rawTx.slice(this.rawTx.length - 4, this.rawTx.length)]));
      } else {
        this.txId = window.sha256(this.rawTx);
        this.hash = this.txId;
        this.segwit = false;
      }
      if (!A.keepRawTx) this.rawTx = null;
      if (A.format === 'decoded') this.decode();
    }
  }


  /**
  * change Transaction object representation to "decoded" human readable format
  *
  * :param testnet: (optional) flag for testnet network, by default is ``false``.
  */
  Transaction.prototype.decode = function (testnet) {
    this.format = 'decoded';
    if (testnet !== undefined) this.testnet = testnet;
    if (isBuffer(this.txId)) this.txId = rh2s(this.txId);
    if (isBuffer(this.hash)) this.hash = rh2s(this.hash);
    if (isBuffer(this.flag)) this.flag = rh2s(this.flag);
    if (isBuffer(this.rawTx)) this.rawTx = this.rawTx.hex();
    for (let i in this.vIn) {
      if (isBuffer(this.vIn[i].txId)) this.vIn[i].txId = rh2s(this.vIn[i].txId);
      if (isBuffer(this.vIn[i].scriptSig)) this.vIn[i].scriptSig = this.vIn[i].scriptSig.hex();
      if (this.vIn[i].amount instanceof window.BN) this.vIn[i].amount = this.vIn[i].amount.toString(16);
      if (this.vIn[i].txInWitness !== undefined) {
        let t = [];
        for (let w of this.vIn[i].txInWitness) t.push((isBuffer(w) ? w.hex() : w));
        this.vIn[i].txInWitness = t;
      }
      if (isBuffer(this.vIn[i].addressHash)) {
        let w = (this.vIn[i].nType < 5) ? null : this.vIn[i].addressHash[0];
        this.vIn[i].addressHash = this.vIn[i].addressHash.hex();
        let sh = [1, 5].includes(this.vIn[i].nType);
        this.vIn[i].address = window.hashToAddress(this.vIn[i].addressHash,
          { testnet: this.testnet, scriptHash: sh, witnessVersion: w });

      }
      if (isBuffer(this.vIn[i].scriptPubKey)) {
        this.vIn[i].scriptPubKey = this.vIn[i].scriptPubKey.hex();
        this.vIn[i].scriptPubKeyOpcodes = window.decodeScript(this.vIn[i].scriptPubKey);
        this.vIn[i].scriptPubKeyAsm = window.decodeScript(this.vIn[i].scriptPubKey, { asm: true });
      }
      if (isBuffer(this.vIn[i].redeemScript)) {
        this.vIn[i].redeemScript = this.vIn[i].redeemScript.hex();
        this.vIn[i].redeemScriptOpcodes = window.decodeScript(this.vIn[i].redeemScript);
        this.vIn[i].redeemScriptAsm = window.decodeScript(this.vIn[i].redeemScript, { asm: true });
      }
      if (!this.coinbase) {
        if (isBuffer(this.vIn[i].scriptSig)) {
          this.vIn[i].scriptSig = this.vIn[i].scriptSig.hex();
        }

        this.vIn[i].scriptSigOpcodes = window.decodeScript(this.vIn[i].scriptSig);
        this.vIn[i].scriptSigAsm = window.decodeScript(this.vIn[i].scriptSig, { asm: true });
      }
    }

    for (let i in this.vOut) {
      if (isBuffer(this.vOut[i].addressHash)) {
        let w = (this.vOut[i].nType < 5) ? null : this.vOut[i].scriptPubKey[0];
        this.vOut[i].addressHash = this.vOut[i].addressHash.hex();
        let sh = [1, 5].includes(this.vOut[i].nType);
        this.vOut[i].address = window.hashToAddress(this.vOut[i].addressHash,
          { testnet: this.testnet, scriptHash: sh, witnessVersion: w });

      }
      if (isBuffer(this.vOut[i].scriptPubKey)) {
        this.vOut[i].scriptPubKey = this.vOut[i].scriptPubKey.hex();
        this.vOut[i].scriptPubKeyOpcodes = window.decodeScript(this.vOut[i].scriptPubKey);
        this.vOut[i].scriptPubKeyAsm = window.decodeScript(this.vOut[i].scriptPubKey, { asm: true });
      }


    }
    if (isBuffer(this.data)) this.data = this.data.hex();
    return this;
  };

  /**
  * change Transaction object representation to "raw" bytes format, all human readable part will be stripped.
  */
  Transaction.prototype.encode = function () {
    if (iS(this.txId)) this.txId = s2rh(this.txId);
    if (iS(this.flag)) this.flag = s2rh(this.flag);
    if (iS(this.hash)) this.hash = s2rh(this.hash);
    if (iS(this.rawTx)) this.rawTx = BF(this.hash, 'hex');
    for (let i in this.vIn) {
      if (iS(this.vIn[i].txId)) this.vIn[i].txId = s2rh(this.vIn[i].txId);
      if (iS(this.vIn[i].scriptSig)) this.vIn[i].scriptSig = BF(this.vIn[i].scriptSig, 'hex');
      if (this.vIn[i].txInWitness !== undefined) {
        let t = [];
        for (let w of this.vIn[i].txInWitness) t.push((iS(w) ? BF(w, 'hex') : w));
        this.vIn[i].txInWitness = t;
      }
      if (iS(this.vIn[i].addressHash)) this.vIn[i].addressHash = BF(this.vIn[i].addressHash, 'hex');
      if (iS(this.vIn[i].scriptPubKey)) this.vIn[i].scriptPubKey = BF(this.vIn[i].scriptPubKey, 'hex');
      if (iS(this.vIn[i].redeemScript)) this.vIn[i].redeemScript = BF(this.vIn[i].redeemScript, 'hex');
      if (iS(this.vIn[i].addressHash)) this.vIn[i].addressHash = BF(this.vIn[i].addressHash, 'hex');
      delete this.vIn[i].scriptSigAsm;
      delete this.vIn[i].scriptSigOpcodes;
      delete this.vIn[i].scriptPubKeyOpcodes;
      delete this.vIn[i].scriptPubKeyAsm;
      delete this.vIn[i].redeemScriptOpcodes;
      delete this.vIn[i].redeemScriptAsm;
      delete this.vIn[i].address;
    }
    for (let i in this.vOut) {
      if (iS(this.vOut[i].scriptPubKey)) this.vOut[i].scriptPubKey = BF(this.vOut[i].scriptPubKey, 'hex');
      if (iS(this.vOut[i].addressHash)) this.vOut[i].addressHash = BF(this.vOut[i].addressHash, 'hex');
      delete this.address;
      delete this.vOut[i].scriptPubKeyOpcodes;
      delete this.vOut[i].scriptPubKeyAsm;
    }
    if (iS(this.data)) this.data = BF(this.data, 'hex');
    this.format = 'raw';
    return this;
  };

  /**
  * get serialized Transaction
  *
  * :param segwit: (optional) flag for segwit representation of serialized transaction, by default ``true``.
  * :param hex: (optional) if set to True return HEX encoded string, by default ``true``.
  * :return str,bytes: serialized transaction in HEX or bytes.
  */
  Transaction.prototype.serialize = function (A = {}) {
    ARGS(A, { segwit: true, hex: true });
    let chunks = [];
    chunks.push(BF(window.intToBytes(this.version, 4)));
    if (A.segwit && this.segwit) chunks.push(BF([0, 1]));
    chunks.push(BF(window.intToVarInt(Object.keys(this.vIn).length)));

    for (let i in this.vIn) {
      if (iS(this.vIn[i].txId)) chunks.push(s2rh(this.vIn[i].txId));
      else chunks.push(this.vIn[i].txId);
      chunks.push(BF(window.intToBytes(this.vIn[i].vOut, 4)));
      let s = (iS(this.vIn[i].scriptSig)) ? BF(this.vIn[i].scriptSig, 'hex') : this.vIn[i].scriptSig;

      chunks.push(BF(window.intToVarInt(s.length)));
      chunks.push(s);
      chunks.push(BF(window.intToBytes(this.vIn[i].sequence, 4)));
    }
    chunks.push(BF(window.intToVarInt(Object.keys(this.vOut).length)));

    for (let i in this.vOut) {
      chunks.push(BF(window.intToBytes(this.vOut[i].value, 8)));
      let s = (iS(this.vOut[i].scriptPubKey)) ? BF(this.vOut[i].scriptPubKey, 'hex') : this.vOut[i].scriptPubKey;
      chunks.push(BF(window.intToVarInt(s.length)));
      chunks.push(s);
    }
    if (A.segwit && this.segwit) {
      for (let i in this.vIn) {
        chunks.push(BF(window.intToVarInt(this.vIn[i].txInWitness.length)));
        for (let w of this.vIn[i].txInWitness) {
          let s = iS(w) ? BF(w, 'hex') : w;
          chunks.push(BF(window.intToVarInt(s.length)));
          chunks.push(s);
        }
      }
    }
    chunks.push(BF(window.intToBytes(this.lockTime, 4)));
    let out = BC(chunks);
    return (A.hex) ? out.hex() : out;
  };

  /**
  * get json Transaction representation
  */
  Transaction.prototype.json = function () {
    let r;
    if (this.format === 'raw') {
      this.decode();
      r = JSON.stringify(this);
      this.encode();
    } else r = JSON.stringify(this);
    return r;
  };

  Transaction.prototype.addInput = function (A = {}) {
    ARGS(A, {
      txId: null, vOut: 0, sequence: 0xffffffff,
      scriptSig: "", txInWitness: null, value: null,
      scriptPubKey: null, address: null, privateKey: null,
      redeemScript: null, inputVerify: true
    });
    let witness = [], s;
    if (A.txId === null) {
      A.txId = Buffer(32);
      A.vOut = 0xffffffff;
      if (((A.sequence !== 0xffffffff) || (Object.keys(this.vOut).length)) && (A.inputVerify))
        throw new Error('invalid coinbase transaction');
    }
    if (iS(A.txId))
      if (window.isHex(A.txId)) A.txId = s2rh(A.txId);
      else throw new Error('txId invalid');
    if (!isBuffer(A.txId) || A.txId.length !== 32) throw new Error('txId invalid');

    if (A.scriptSig.length === 0) A.scriptSig = BF([]);
    if (iS(A.scriptSig))
      if (window.isHex(A.scriptSig)) A.scriptSig = BF(A.scriptSig, 'hex');
      else throw new Error('scriptSig invalid');
    if (!isBuffer(A.scriptSig) || ((A.scriptSig.length > 520) && (A.inputVerify)))
      throw new Error('scriptSig invalid');

    if ((A.vOut < 0) || A.vOut > 0xffffffff) throw new Error('vOut invalid');
    if ((A.sequence < 0) || A.sequence > 0xffffffff) throw new Error('vOut invalid');

    if ((A.privateKey !== null) && (!(A.privateKey instanceof window.PrivateKey)))
      A.privateKey = window.PrivateKey(A.privateKey);

    if ((A.value !== null) && ((A.value < 0) || (A.value > window.MAX_AMOUNT)))
      throw new Error('amount invalid');

    if (A.txInWitness !== null) {
      let l = 0;
      for (let w of A.txInWitness) {
        if (iS(w)) witness.push((this.format === 'raw') ? BF(w, 'hex') : w);
        else witness.push((this.format === 'raw') ? w : BF(w, 'hex'));
        l += 1 + w.length;
      }
    }

    if (A.txId.equals(Buffer.alloc(32))) {
      if (!((A.vOut === 0xffffffff) && (A.sequence === 0xffffffff) && (A.scriptSig.length <= 100)))
        if (A.inputVerify) throw new Error("coinbase tx invalid");
      this.coinbase = true;
    }

    if (A.scriptPubKey !== null) {
      if (iS(A.scriptPubKey)) A.scriptPubKey = BF(A.scriptPubKey, 'hex');
      if (!isBuffer(A.scriptPubKey)) throw new Error("scriptPubKey invalid");
    }

    if (A.redeemScript !== null) {
      if (iS(A.redeemScript)) A.redeemScript = BF(A.redeemScript, 'hex');
      if (!isBuffer(A.redeemScript)) throw new Error("scriptPubKey invalid");
    }

    if (A.address !== null) {
      if (iS(A.address)) {
        let net = window.addressNetType(A.address) === 'mainnet';
        if (!(net !== this.testnet)) throw new Error("address invalid");
        s = window.addressToScript(A.address);
      } else if (A.address.address !== undefined) s = window.addressToScript(A.address.address);
      else throw new Error("address invalid");
      if (A.scriptPubKey !== null) {
        if (!A.scriptPubKey.equals(s)) throw new Error("address not match script");
      } else A.scriptPubKey = s;

    }

    let k = Object.keys(this.vIn).length;
    this.vIn[k] = {};
    this.vIn[k].vOut = A.vOut;
    this.vIn[k].sequence = A.sequence;
    if (this.format === 'raw') {
      this.vIn[k].txId = A.txId;
      this.vIn[k].scriptSig = A.scriptSig;
      if (A.scriptPubKey !== null) this.vIn[k].scriptPubKey = A.scriptPubKey;
      if (A.redeemScript !== null) this.vIn[k].redeemScript = A.redeemScript;
    } else {
      this.vIn[k].txId = rh2s(A.txId);
      this.vIn[k].scriptSig = A.scriptSig.hex();
      this.vIn[k].scriptSigOpcodes = window.decodeScript(A.scriptSig);
      this.vIn[k].scriptSigAsm = window.decodeScript(A.scriptSig, { asm: true });
      if (A.scriptPubKey !== null) {
        this.vIn[k].scriptPubKey = A.scriptPubKey.hex();
        this.vIn[k].scriptPubKeyOpcodes = window.decodeScript(A.scriptPubKey);
        this.vIn[k].scriptPubKeyAsm = window.decodeScript(A.scriptPubKey, { asm: true });
      }
      if (A.redeemScript !== null) {
        this.vIn[k].redeemScript = A.redeemScript.hex();
        this.vIn[k].redeemScriptOpcodes = window.decodeScript(A.redeemScript);
        this.vIn[k].redeemScriptAsm = window.decodeScript(A.redeemScript, { asm: true });
      }
    }

    if (A.txInWitness !== null) {
      this.segwit = true;
      this.vIn[k].txInWitness = witness;
    }
    if (A.value !== null) this.vIn[k].value = A.value;
    if (A.privateKey !== 0) this.vIn[k].privateKey = A.privateKey;
    if (this.autoCommit) this.commit();
    return this;
  };

  Transaction.prototype.addOutput = function (A = {}) {
    ARGS(A, { value: 0, address: null, scriptPubKey: null });
    if ((A.address === null) && (A.scriptPubKey === null))
      throw new Error("unable to add output, address or script required");
    if ((A.value < 0) || (A.value > window.MAX_AMOUNT)) throw new Error(" amount value error");
    if (A.scriptPubKey !== null)
      if (iS(A.scriptPubKey)) A.scriptPubKey = BF(A.scriptPubKey, 'hex');
      else if (A.address !== null)
        if (A.address.address !== undefined) A.address = A.address.address;
    if (A.address !== null)
      A.scriptPubKey = window.addressToScript(A.address);


    let k = Object.keys(this.vOut).length;
    this.vOut[k] = {};
    this.vOut[k].value = A.value;

    let s = window.parseScript(A.scriptPubKey, { segwit: true });
    this.vOut[k].nType = s.nType;
    this.vOut[k].type = s.type;

    if (this.format === 'raw') {
      this.vOut[k].scriptPubKey = A.scriptPubKey;
      if ((this.data === null) && (s.nType === 3)) this.data = s.data;
      if (!([3, 4, 7, 8].includes(s.nType))) {
        this.vOut[k].addressHash = s.addressHash;
        this.vOut[k].reqSigs = s.reqSigs;
      }
    } else {
      this.vOut[k].scriptPubKey = A.scriptPubKey.hex();
      if ((this.data === null) && (s.nType === 3)) this.data = s.data.hex();
      if (!([3, 4, 7, 8].includes(s.nType))) {
        this.vOut[k].addressHash = s.addressHash.hex();
        this.vOut[k].reqSigs = s.reqSigs;
      }
      this.vOut[k].scriptPubKeyOpcodes = window.decodeScript(A.scriptPubKey);
      this.vOut[k].scriptPubKeyAsm = window.decodeScript(A.scriptPubKey, { "asm": true });
      let sh = [1, 5].includes(s.nType);
      let witnessVersion = (s.nType < 5) ? null : A.scriptPubKey[0];
      if (this.vOut[k].addressHash !== undefined)
        this.vOut[k].address = window.hashToAddress(this.vOut[k].addressHash,
          { testnet: this.testnet, scriptHash: sh, witnessVersion: witnessVersion });
    }
    if (this.autoCommit) this.commit();
    return this;
  };

  Transaction.prototype.delOutput = function (n) {
    let l = Object.keys(this.vOut).length;
    if (l === 0) return this;
    if (n === undefined) n = l - 1;
    let out = {};
    let c = 0;
    for (let i = 0; i < l; i++) {
      if (i !== n) {
        out[c] = this.vOut[i];
        c++;
      }
    }
    this.vOut = out;
    if (this.autoCommit) this.commit();
    return this;
  };

  Transaction.prototype.delInput = function (n) {
    let l = Object.keys(this.vIn).length;
    if (l === 0) return this;
    if (n === undefined) n = l - 1;
    let out = {};
    let c = 0;
    for (let i = 0; i < l; i++) {
      if (i !== n) {
        out[c] = this.vIn[i];
        c++;
      }
    }
    this.vOut = out;
    if (this.autoCommit) this.commit();
    return this;
  };

  Transaction.prototype.commit = function () {
    if ((Object.keys(this.vIn).length === 0) || (Object.keys(this.vOut).length === 0)) return this;

    if (this.segwit)
      for (let i in this.vIn) if (this.vIn[i].txInWitness === undefined) this.vIn[i].txInWitness = [];
    let nonSegwitView = this.serialize({ segwit: false, hex: false });
    this.txId = window.sha256(nonSegwitView);
    this.rawTx = this.serialize({ segwit: true, hex: false });
    this.hash = window.sha256(this.rawTx);
    this.size = this.rawTx.length;
    this.bSize = nonSegwitView.length;
    this.weight = this.bSize * 3 + this.size;
    this.vSize = Math.ceil(this.weight / 4);

    if (this.format !== 'raw') {
      this.txId = rh2s(this.txId);
      this.hash = rh2s(this.hash);
      this.rawTx = this.rawTx.hex();
    }
    let inputSum = 0;
    let outputSum = 0;
    for (let i in this.vIn) {
      if (this.vIn[i].value !== undefined) inputSum += this.vIn[i].value;
      else {
        inputSum = null;
        break
      }

      for (let i in this.vOut)
        if (this.vOut[i].value !== undefined) outputSum += this.vOut[i].value;
    }
    this.amount = outputSum;
    if (outputSum && inputSum) this.fee = inputSum - outputSum;
    else this.fee = null;
    return this;
  };

  Transaction.prototype.sigHash = function (n, A = {}) {
    ARGS(A, { scriptPubKey: null, sigHashType: window.SIGHASH_ALL, preImage: false });
    if (this.vIn[n] === undefined) throw new Error("input not exist");
    let scriptCode;
    if (A.scriptPubKey !== null) scriptCode = A.scriptPubKey;
    else {
      if (this.vIn[n].scriptPubKey === undefined) throw new Error("scriptPubKey required");
      scriptCode = this.vIn[n].scriptPubKey;
    }
    scriptCode = getBuffer(scriptCode);

    if (((A.sigHashType & 31) === window.SIGHASH_SINGLE) && (n >= Object.keys(this.vOut).length)) {
      let r = BC([BF([1]), BA(31)]);
      return (this.format === 'raw') ? r : rh2s(r);
    }

    scriptCode = window.deleteFromScript(scriptCode, BF([O.OP_CODESEPARATOR]));
    let pm = [BF(window.intToBytes(this.version, 4))];
    pm.push((A.sigHashType & window.SIGHASH_ANYONECANPAY) ? BF([1]) : BF(window.intToVarInt(Object.keys(this.vIn).length)));

    for (let i in this.vIn) {
      i = parseInt(i);
      if ((A.sigHashType & window.SIGHASH_ANYONECANPAY) && (n !== i)) continue;
      let sequence = this.vIn[i].sequence;
      if (([window.SIGHASH_SINGLE, window.SIGHASH_NONE].includes(A.sigHashType & 31)) && (n !== i)) sequence = 0;
      let txId = iS(this.vIn[i].txId) ? s2rh(this.vIn[i].txId) : this.vIn[i].txId;
      pm.push(txId);
      pm.push(BF(window.intToBytes(this.vIn[i].vOut, 4)));

      if (n === i) {
        pm.push(BF(window.intToVarInt(scriptCode.length)));
        pm.push(scriptCode);
        pm.push(BF(window.intToBytes(sequence, 4)));
      } else {
        pm.push(BF([0]));
        pm.push(BF(window.intToBytes(sequence, 4)));
      }
    }

    if ((A.sigHashType & 31) === window.SIGHASH_NONE) pm.push(BF([0]));
    else if ((A.sigHashType & 31) === window.SIGHASH_SINGLE) pm.push(BF(window.intToVarInt(n + 1)));
    else pm.push(BF(window.intToVarInt(Object.keys(this.vOut).length)));
    let scriptPubKey;

    if ((A.sigHashType & 31) !== window.SIGHASH_NONE) {
      for (let i in this.vOut) {
        i = parseInt(i);
        scriptPubKey = this.vOut[i].scriptPubKey;
        scriptPubKey = iS(scriptPubKey) ? BF(scriptPubKey, 'hex') : scriptPubKey;

        if ((i > n) && ((A.sigHashType & 31) === window.SIGHASH_SINGLE)) continue;
        if (((A.sigHashType & 31) === window.SIGHASH_SINGLE) && (n !== i)) {
          pm.push(BA(8, 0xff));
          pm.push(BA(1, 0x00));
        } else {
          pm.push(BF(window.intToBytes(this.vOut[i].value, 8)));
          pm.push(BF(window.intToVarInt(scriptPubKey.length)));
          pm.push(scriptPubKey);
        }
      }
    }
    pm.push(BF(window.intToBytes(this.lockTime, 4)));
    pm.push(BF(window.intToBytes(A.sigHashType, 4)));
    pm = BC(pm);
    if (!A.preImage) {
      pm = window.doubleSha256(pm);
      return (this.format === 'raw') ? pm : rh2s(pm);
    }
    return (this.format === 'raw') ? pm : pm.hex();
  };

  Transaction.prototype.sigHashSegwit = function (n, A = {}) {
    ARGS(A, { value: null, scriptPubKey: null, sigHashType: window.SIGHASH_ALL, preImage: false });
    if (this.vIn[n] === undefined) throw new Error("input not exist");
    let scriptCode, value;

    if (A.scriptPubKey !== null) scriptCode = A.scriptPubKey;
    else {
      if (this.vIn[n].scriptPubKey === undefined) throw new Error("scriptPubKey required");
      scriptCode = this.vIn[n].scriptPubKey;
    }
    scriptCode = getBuffer(scriptCode);

    if (A.value !== null) value = A.value;
    else {
      if (this.vIn[n].value === undefined) throw new Error("value required");
      value = this.vIn[n].value;
    }

    let hp = [], hs = [], ho = [], outpoint, nSequence;

    for (let i in this.vIn) {
      i = parseInt(i);
      let txId = this.vIn[i].txId;
      if (iS(txId)) txId = s2rh(txId);

      let vOut = BF(window.intToBytes(this.vIn[i].vOut, 4));
      if (!(A.sigHashType & window.SIGHASH_ANYONECANPAY)) {
        hp.push(txId);
        hp.push(vOut);
        if (((A.sigHashType & 31) !== window.SIGHASH_SINGLE) && ((A.sigHashType & 31) !== window.SIGHASH_NONE))
          hs.push(BF(window.intToBytes(this.vIn[i].sequence, 4)));
      }
      if (i === n) {
        outpoint = BC([txId, vOut]);
        nSequence = BF(window.intToBytes(this.vIn[i].sequence, 4));
      }
    }
    // https://github.com/wu-emma/bitgesell/blob/cb9f0da214f38691b0a4947fd9f9c4ff9a647f43/src/script/interpreter.cpp#L1186
    let hashPrevouts = (hp.length > 0) ? window.sha3(BC(hp)) : BA(32, 0);
    // https://github.com/wu-emma/bitgesell/blob/cb9f0da214f38691b0a4947fd9f9c4ff9a647f43/src/script/interpreter.cpp#L1196
    let hashSequence = (hs.length > 0) ? window.sha3(BC(hs)) : BA(32, 0);
    value = BF(window.intToBytes(value, 8));

    for (let o in this.vOut) {
      o = parseInt(o);
      let scriptPubKey = getBuffer(this.vOut[o].scriptPubKey);
      if (!([window.SIGHASH_SINGLE, window.SIGHASH_NONE].includes(A.sigHashType & 31))) {
        ho.push(BF(window.intToBytes(this.vOut[o].value, 8)));
        ho.push(BF(window.intToVarInt(scriptPubKey.length)));
        ho.push(scriptPubKey);
      } else if (((A.sigHashType & 31) === window.SIGHASH_SINGLE) && (n < Object.keys(this.vOut).length)) {
        if (o === n) {
          ho.push(BF(window.intToBytes(this.vOut[o].value, 8)));
          ho.push(BF(window.intToVarInt(scriptPubKey.length)));
          ho.push(scriptPubKey);
        }
      }
    }
    // https://github.com/wu-emma/bitgesell/blob/cb9f0da214f38691b0a4947fd9f9c4ff9a647f43/src/script/interpreter.cpp#L1206
    let hashOutputs = (ho.length > 0) ? window.sha3(BC(ho)) : BA(32, 0);
    let pm = BC([BF(window.intToBytes(this.version, 4)),
      hashPrevouts, hashSequence, outpoint, scriptCode,
      value, nSequence, hashOutputs,
    BF(window.intToBytes(this.lockTime, 4)),
    BF(window.intToBytes(A.sigHashType, 4))]);

    if (A.preImage) return (this.format === 'raw') ? pm.hex() : pm;
    // https://github.com/wu-emma/bitgesell/blob/cb9f0da214f38691b0a4947fd9f9c4ff9a647f43/src/hash.h#L180
    // https://github.com/wu-emma/bitgesell/blob/cb9f0da214f38691b0a4947fd9f9c4ff9a647f43/src/script/interpreter.cpp#L1281
    return window.sha3(pm, { 'hex': this.format !== 'raw' });

  };

  Transaction.prototype.signInput = function (n, A = {}) {
    ARGS(A, {
      privateKey: null, scriptPubKey: null, redeemScript: null,
      sigHashType: window.SIGHASH_ALL, address: null, value: null,
      witnessVersion: 0, p2sh_p2wsh: false
    });
    if (this.vIn[n] === undefined) throw new Error('input not exist');
    // privateKey
    if (A.privateKey === null) {
      if (this.vIn[n].privateKey === undefined) throw new Error('no private key');
      A.privateKey = this.vIn[n].privateKey;
    }

    if (A.privateKey instanceof Array) {
      A.publicKey = [];
      let pk = [];
      for (let key of A.privateKey) {
        if (key.key !== undefined) key = key.wif;
        A.publicKey.push(window.privateToPublicKey(key, { hex: false }));
        pk.push(new window.PrivateKey(key).key);
      }
      A.privateKey = pk;
    } else {
      if (A.privateKey.key === undefined) {
        let k = new window.PrivateKey(A.privateKey);
        A.privateKey = k.key;
        A.privateKeyCompressed = k.compressed;
      } else {
        A.privateKeyCompressed = A.privateKey.compressed;
        A.privateKey = A.privateKey.key;
      }
      A.publicKey = [window.privateToPublicKey(A.privateKey, { hex: false, compressed: A.privateKeyCompressed })];
      A.privateKey = [A.privateKey];
    }

    // address

    if ((A.address === null) && (this.vIn[n].address !== undefined)) A.address = this.vIn[n].address;
    if (A.address !== null) {
      if (A.address.address !== undefined) A.address = A.address.address;

      if (this.testnet !== (window.addressNetType(A.address) === 'testnet'))
        throw new Error('address network invalid');
      A.scriptPubKey = window.addressToScript(A.address);
    }

    let scriptType = null;

    // redeem script
    if ((A.redeemScript === null) && (this.vIn[n].redeemScript !== undefined)) A.redeemScript = this.vIn[n].redeemScript;
    if (A.redeemScript !== null) A.redeemScript = getBuffer(A.redeemScript);

    // script pub key
    if ((A.scriptPubKey === null) && (this.vIn[n].scriptPubKey !== undefined))
      A.scriptPubKey = this.vIn[n].scriptPubKey;
    else if ((A.scriptPubKey === null) && (A.redeemScript === null)) throw new Error('no scriptPubKey key');


    if (A.scriptPubKey !== null) {
      A.scriptPubKey = getBuffer(A.scriptPubKey);

      let p = window.parseScript(A.scriptPubKey);
      scriptType = p.type;
      if ([5, 6].includes(p.nType)) A.witnessVersion = A.scriptPubKey[0];
    } else if (A.redeemScript !== null) {
      if ((A.witnessVersion === null) || (A.p2sh_p2wsh)) scriptType = "P2SH";
      else scriptType = "P2WSH";
    }

    // sign input
    let sigSript;
    switch (scriptType) {
      case 'PUBKEY':
        sigSript = this.__sign_PUBKEY(n, A);
        break;
      case 'P2PKH':
        sigSript = this.__sign_P2PKH(n, A);
        break;
      case 'P2SH':
        sigSript = this.__sign_P2SH(n, A);
        break;
      case 'P2WPKH':
        sigSript = this.__sign_P2WPKH(n, A);
        break;
      case 'P2WSH':
        sigSript = this.__sign_P2WSH(n, A);
        break;
      case 'MULTISIG':
        sigSript = this.__sign_MULTISIG(n, A);
        break;
      default:
        throw new Error('not implemented');
    }

    if (this.format === 'raw') this.vIn[n].scriptSig = sigSript;
    else {
      this.vIn[n].scriptSig = sigSript.hex();
      this.vIn[n].scriptSigOpcodes = window.decodeScript(sigSript);
      this.vIn[n].scriptSigAsm = window.decodeScript(sigSript, { asm: true });
    }
    if (this.autoCommit) this.commit();
    return this;
  };

  Transaction.prototype.__sign_PUBKEY = function (n, A) {
    let sighash = this.sigHash(n, A);
    if (iS(sighash)) sighash = s2rh(sighash);

    let signature = BC([window.signMessage(sighash, A.privateKey[0]).signature,
    BF(window.intToBytes(A.sigHashType, 1))]);
    if (this.format === 'raw') this.vIn[n].signatures = [signature];
    else this.vIn[n].signatures = [signature.hex()];
    return BC([BF([signature.length]), signature]);
  };

  Transaction.prototype.__sign_P2PKH = function (n, A) {
    let sighash = this.sigHash(n, A);
    if (iS(sighash)) sighash = s2rh(sighash);
    let signature = BC([window.signMessage(sighash, A.privateKey[0]).signature, BF(window.intToBytes(A.sigHashType, 1))]);
    if (this.format === 'raw') this.vIn[n].signatures = [signature];
    else this.vIn[n].signatures = [signature.hex()];
    return BC([BF([signature.length]), signature, BF([A.publicKey[0].length]), A.publicKey[0]]);
  };

  Transaction.prototype.__sign_P2SH = function (n, A) {
    if (A.redeemScript === null) throw new Error('no redeem script');
    if (A.p2sh_p2wsh) return this.__sign_P2SH_P2WSH(n, A);
    let scriptType = window.parseScript(A.redeemScript)["type"];
    switch (scriptType) {
      case 'MULTISIG':
        return this.__sign_P2SH_MULTISIG(n, A);
      case 'P2WPKH':
        return this.__sign_P2SH_P2WPKH(n, A);
      default:
        throw new Error('not implemented');
    }
  };

  Transaction.prototype.__sign_P2SH_MULTISIG = function (n, A) {
    let sighash = this.sigHash(n, { scriptPubKey: A.redeemScript, sigHashType: A.sigHashType });

    if (iS(sighash)) sighash = s2rh(sighash);
    let sig = [];
    this.vIn[n].signatures = [];
    for (let key of A.privateKey) {
      let s = BC([window.signMessage(sighash, key).signature, BF(window.intToBytes(A.sigHashType, 1))]);
      sig.push(s);
      this.vIn[n].signatures.push((this.format === 'raw') ? s : s.hex());
    }
    return this.__get_MULTISIG_scriptSig(n, A.publicKey,
      sig, A.redeemScript, A.redeemScript);
  };

  Transaction.prototype.__sign_P2SH_P2WPKH = function (n, A) {
    let s = BC([BF([0x19]), BF([O.OP_DUP, O.OP_HASH160]),
    window.opPushData(window.hash160(A.publicKey[0])), BF([O.OP_EQUALVERIFY, O.OP_CHECKSIG])]);
    if (A.value === null) {
      if (this.vIn[n].value !== undefined) A.value = this.vIn[n].value;
      else throw new Error('no input amount');
    }

    let sighash = this.sigHashSegwit(n, { scriptPubKey: s, sigHashType: A.sigHashType, value: A.value });
    sighash = getBuffer(sighash);
    let signature = BC([window.signMessage(sighash, A.privateKey[0]).signature, BF(window.intToBytes(A.sigHashType, 1))]);
    this.segwit = true;
    if (this.format === 'raw') this.vIn[n].txInWitness = [signature, A.publicKey[0]];
    else this.vIn[n].txInWitness = [signature.hex(), A.publicKey[0].hex()];
    this.vIn[n].signatures = (this.format === 'raw') ? [signature] : [signature.hex()];
    return window.opPushData(A.redeemScript);
  };

  Transaction.prototype.__sign_P2SH_P2WSH = function (n, A) {
    let scriptType = window.parseScript(A.redeemScript)["type"];
    switch (scriptType) {
      case 'MULTISIG':
        return this.__sign_P2SH_P2WSH_MULTISIG(n, A);
      default:
        throw new Error('not implemented');
    }
  };

  Transaction.prototype.__sign_P2SH_P2WSH_MULTISIG = function (n, A) {
    this.segwit = true;
    let scriptCode = BC([BF(window.intToVarInt(A.redeemScript.length)), A.redeemScript]);
    let sighash = this.sigHashSegwit(n, {
      scriptPubKey: scriptCode,
      sigHashType: A.sigHashType, value: A.value
    });
    sighash = getBuffer(sighash);
    this.vIn[n].signatures = [];
    let sig = [];
    for (let key of A.privateKey) {
      let s = BC([window.signMessage(sighash, key).signature, BF(window.intToBytes(A.sigHashType, 1))]);
      sig.push(s);
      this.vIn[n].signatures.push((this.format === 'raw') ? s : s.hex());
    }
    let witness = this.__get_MULTISIG_scriptSig(n, A.publicKey, sig, scriptCode, A.redeemScript, A.value);
    if (this.format === 'raw') this.vIn[n].txInWitness = witness;
    else {
      this.vIn[n].txInWitness = [];
      for (let w of witness) this.vIn[n].txInWitness.push(w.hex());
    }
    // calculate P2SH redeem script from P2WSH redeem script
    return window.opPushData(BC([BF([0]), window.opPushData(window.sha256(A.redeemScript))]))
  };

  Transaction.prototype.__sign_P2WPKH = function (n, A) {
    let s = BC([BF([0x19]), BF([O.OP_DUP, O.OP_HASH160]), A.scriptPubKey.slice(1),
    BF([O.OP_EQUALVERIFY, O.OP_CHECKSIG])]);
    if (A.value === null) {
      if (this.vIn[n].value !== undefined) A.value = this.vIn[n].value;
      else throw new Error('no input amount');
    }
    let sighash = this.sigHashSegwit(n, { scriptPubKey: s, sigHashType: A.sigHashType, value: A.value });
    sighash = getBuffer(sighash);
    let signature = BC([window.signMessage(sighash, A.privateKey[0]).signature, BF(window.intToBytes(A.sigHashType, 1))]);
    this.segwit = true;
    if (this.format === 'raw') this.vIn[n].txInWitness = [signature, A.publicKey[0]];
    else this.vIn[n].txInWitness = [signature.hex(), A.publicKey[0].hex()];
    this.vIn[n].signatures = (this.format === 'raw') ? [signature] : [signature.hex()];
    return BF([]);
  };

  Transaction.prototype.__sign_P2WSH = function (n, A) {
    this.segwit = true;
    if (A.value === null) {
      if (this.vIn[n].value !== undefined) A.value = this.vIn[n].value;
      else throw new Error('no input amount');
    }
    let scriptType = window.parseScript(A.redeemScript)["type"];
    switch (scriptType) {
      case 'MULTISIG':
        return this.__sign_P2WSH_MULTISIG(n, A);
      default:
        throw new Error('not implemented');
    }
  };

  Transaction.prototype.__sign_P2WSH_MULTISIG = function (n, A) {
    let scriptCode = BC([BF(window.intToVarInt(A.redeemScript.length)), A.redeemScript]);

    let sighash = this.sigHashSegwit(n, {
      scriptPubKey: scriptCode,
      sigHashType: A.sigHashType, value: A.value
    });
    sighash = getBuffer(sighash);
    let sig = [];
    this.vIn[n].signatures = [];
    for (let key of A.privateKey) {
      let s = BC([window.signMessage(sighash, key).signature, BF(window.intToBytes(A.sigHashType, 1))]);
      sig.push(s);
      this.vIn[n].signatures.push((this.format === 'raw') ? s : s.hex());
    }

    let witness = this.__get_MULTISIG_scriptSig(n, A.publicKey, sig, scriptCode, A.redeemScript, A.value);
    if (this.format === 'raw') this.vIn[n].txInWitness = witness;
    else {
      this.vIn[n].txInWitness = [];
      for (let w of witness) this.vIn[n].txInWitness.push(w.hex());
    }
    return BF([]);
  };

  Transaction.prototype.__sign_MULTISIG = function (n, A) {
    let sighash = this.sigHash(n, { scriptPubKey: A.scriptPubKey, sigHashType: A.sigHashType });
    if (iS(sighash)) sighash = s2rh(sighash);
    let sig = [];
    this.vIn[n].signatures = [];
    for (let key of A.privateKey) {
      let s = BC([window.signMessage(sighash, key).signature, BF(window.intToBytes(A.sigHashType, 1))]);
      sig.push(s);
      this.vIn[n].signatures.push((this.format === 'raw') ? s : s.hex());
    }
    return this.__get_bare_multisig_script_sig__(n, A.publicKey,
      sig, A.scriptPubKey);
  };

  Transaction.prototype.__get_bare_multisig_script_sig__ = function (n, publicKeys, signatures, scriptPubKey) {
    let sigMap = {};
    for (let i in publicKeys) sigMap[publicKeys[i]] = signatures[i];
    scriptPubKey.seek(0);
    let pubKeys = window.getMultiSigPublicKeys(scriptPubKey);
    let s = getBuffer(this.vIn[n].scriptSig);
    s.seek(0);
    let r = window.readOpCode(s);
    while (r[0] !== null) {
      r = window.readOpCode(s);
      if ((r[1] !== null) && window.isValidSignatureEncoding(r[1])) {
        let sigHash = this.sigHash(n, { scriptPubKey: scriptPubKey, sigHashType: r[1][r[1].length - 1] });
        if (iS(sigHash)) sigHash = s2rh(sigHash);
        for (let i = 0; i < 4; i++) {
          let pk = window.publicKeyRecovery(r[1].slice(0, r[1].length - 1), sigHash, i, { hex: false });
          if (pk === null) continue;
          for (let p of pubKeys)
            if (pk.equals(p)) {
              sigMap[pk] = r[1];
              break;
            }
        }
      }
    }
    r = [BF([O.OP_0])];
    for (let p of pubKeys)
      if (sigMap[p] !== undefined) r.push(window.opPushData(sigMap[p]));
    return BC(r);
  };

  Transaction.prototype.__get_MULTISIG_scriptSig = function (n, publicKeys, signatures, scriptCode,
    redeemScript, value = null) {
    let sigMap = {};
    for (let i in publicKeys) sigMap[publicKeys[i]] = signatures[i];
    redeemScript.seek(0);
    let pubKeys = window.getMultiSigPublicKeys(redeemScript);
    let p2wsh = (value !== null);
    if (!p2wsh) {
      let s = getBuffer(this.vIn[n].scriptSig);
      s.seek(0);
      let r = window.readOpCode(s);
      while (r[0] !== null) {
        r = window.readOpCode(s);
        if ((r[1] !== null) && window.isValidSignatureEncoding(r[1])) {
          let sigHash = this.sigHash(n, { scriptPubKey: scriptCode, sigHashType: r[1][r[1].length - 1] });
          if (iS(sigHash)) sigHash = s2rh(sigHash);
          for (let i = 0; i < 4; i++) {
            let pk = window.publicKeyRecovery(r[1].slice(0, r[1].length - 1), sigHash, i, { hex: false });
            if (pk === null) continue;
            for (let p of pubKeys)
              if (pk.equals(p)) {
                sigMap[pk] = r[1];
                break;
              }
          }
        }
      }
      r = [BF([O.OP_0])];
      for (let p of pubKeys)
        if (sigMap[p] !== undefined) r.push(window.opPushData(sigMap[p]));

      r.push(window.opPushData(redeemScript));
      return BC(r);
    }
    if (this.vIn[n].txInWitness !== undefined)
      for (let w of this.vIn[n].txInWitness) {
        w = getBuffer(w);
        if ((w.length > 0) && (window.isValidSignatureEncoding(w))) {
          let sigHash = this.sigHashSegwit(n, {
            scriptPubKey: scriptCode,
            sigHashType: w[w.length - 1], value: value
          });
          sigHash = getBuffer(sigHash);
          for (let i = 0; i < 4; i++) {
            let pk = window.publicKeyRecovery(w.slice(0, w.length - 1), sigHash, i, { hex: false });
            if (pk === null) continue;
            for (let p of pubKeys)
              if (pk.equals(p)) {
                sigMap[pk] = w;
                break;
              }
          }
        }
      }

    let r = [BF([])];
    for (let p of pubKeys)
      if (sigMap[p] !== undefined) r.push(sigMap[p]);
    r.push(redeemScript);
    return r;
  };



  window.Transaction = Transaction;

}

function Wallet(window) {
  let Buffer = window.Buffer;
  window.defArgs;
  window.getBuffer;
  Buffer.from;
  Buffer.concat;
  window.OPCODE;
  let ARGS = window.defArgs;

  class Wallet {
    /**
   * The class for creating wallet object.
   *
   * :param from: (optional) should be mnemonic phrase, extended public key,extended private key, by default is null (generate new wallet).
   * :param passphrase: (optional) passphrase to get ability use 2FA approach forcreating seed, by default is empty string.
   * :param path: (optional) "BIP44", "BIP49", "BIP84", by default is "BIP84"
   * :param testnet: (optional) flag for testnet network, by default is ``false``.
   * :param strength: (optional) entropy bits strength, by default is 256 bit.
   * :param threshold: (optional) integer, by default is 1
   * :param shares: (optional) integer, by default is 1
   * :param wordList: (optional) word list, by default is BIP39_WORDLIST
   * :param addressType: (optional) "P2PKH", "P2SH_P2WPKH", "P2WPKH"
   * :param hardenedAddresses: (optional) boolean, by default is ``false``.
   * :param account: (optional) integer
   * :param chain: (optional) integer
   */
    constructor(A = {}) {
      ARGS(A, {
        from: null,
        passphrase: "", path: null, testnet: false,
        strength: 256, threshold: 1, shares: 1, wordList: window.BIP39_WORDLIST,
        addressType: null, hardenedAddresses: false, account: 0, chain: 0
      });
      this.account = A.account;
      this.chain = A.chain;
      this.hardenedAddresses = A.hardenedAddresses;
      if (A.path === "BIP84") {
        this.pathType = "BIP84";
        this.path = `m/84'/0'/${this.account}'/${this.chain}`;
        this.__account_path = `m/84'/0'/${this.account}'`;
      } else if (A.path === "BIP49") {
        this.pathType = "BIP49";
        this.path = `m/49'/0'/${this.account}'/${this.chain}`;
        this.__account_path = `m/49'/0'/${this.account}'`;
      } else if (A.path === "BIP44") {
        this.pathType = "BIP44";
        this.path = `m/44'/0'/${this.account}'/${this.chain}`;
        this.__account_path = `m/44'/0'/${this.account}'`;
      } else if (A.path !== null) {
        this.pathType = "custom";
        this.path = A.path;
      } else {
        this.pathType = null;
        this.path = null;
      }
      let from = A.from;
      this.from = from;

      let fromType = null;
      if (from === null) {
        let e = window.generateEntropy({ strength: A.strength });
        /**
        * mnemonic (string)
        */
        this.mnemonic = window.entropyToMnemonic(e, { wordList: A.wordList });
        this.seed = window.mnemonicToSeed(this.mnemonic, {
          hex: true, wordList: A.wordList,
          passphrase: A.passphrase
        });
        this.passphrase = A.passphrase;
        from = window.createMasterXPrivateKey(this.seed, { testnet: A.testnet });
        if (this.pathType === null) {
          this.pathType = "BIP84";
          this.path = `m/84'/0'/${this.account}'/${this.chain}`;
          this.__account_path = `m/84'/0'/${this.account}'`;
        }
        if ((this.pathType !== null) && (this.pathType !== "custom"))
          from = window.BIP32_XKeyToPathXKey(from, this.pathType);

        fromType = "xPriv";
      } else if (window.isString(from)) {

        if (window.isXPrivateKeyValid(from)) {
          if (this.pathType === null) {
            this.pathType = window.xKeyDerivationType(from);
            if (this.pathType === "BIP84") {
              this.path = `m/84'/0'/${this.account}'/${this.chain}`;

              this.__account_path = `m/84'/0'/${this.account}'`;
            } else if (this.pathType === "BIP49") {
              this.path = `m/49'/0'/${this.account}'/${this.chain}`;
              this.__account_path = `m/49'/0'/${this.account}'`;
            } else if (this.pathType === "BIP44") {
              this.path = `m/44'/0'/${this.account}'/${this.chain}`;
              this.__account_path = `m/44'/0'/${this.account}'`;
            } else {
              this.path = "m";
            }
          }

          if ((this.pathType !== null) && (this.pathType !== 'custom'))
            from = window.BIP32_XKeyToPathXKey(window.pathXKeyTo_BIP32_XKey(from), this.pathType);
          fromType = "xPriv";

        } else if (window.isXPublicKeyValid(from)) {

          if (this.pathType === null) {
            this.pathType = window.xKeyDerivationType(from);
            if (this.pathType === "BIP84") {
              this.path = `m/84'/0'/${this.account}'/${this.chain}`;
              this.__account_path = `m/84'/0'/${this.account}'`;
            } else if (this.pathType === "BIP49") {
              this.path = `m/49'/0'/${this.account}'/${this.chain}`;
              this.__account_path = `m/49'/0'/${this.account}'`;
            } else if (this.pathType === "BIP44") {
              this.path = `m/44'/0'/${this.account}'/${this.chain}`;
              this.__account_path = `m/44'/0'/${this.account}'`;
            } else {
              this.path = "m";
            }
          }
          if (this.pathType !== "custom") {
            from = window.BIP32_XKeyToPathXKey(window.pathXKeyTo_BIP32_XKey(from), this.pathType);
            fromType = "xPub";
            if (this.depth === 3) this.__path = "";
          }

        } else {

          if (!window.isMnemonicValid(from, { wordList: A.BIP39_WORDLIST })) throw new Error("invalid mnemonic");

          this.mnemonic = from;
          this.seed = window.mnemonicToSeed(this.mnemonic, {
            hex: true, wordList: A.wordList,
            passphrase: A.passphrase
          });
          this.passphrase = A.passphrase;
          from = window.createMasterXPrivateKey(this.seed, { testnet: A.testnet });

          if (this.pathType === null) {
            this.pathType = "BIP84";
            this.path = `m/84'/0'/${this.account}'/${this.chain}`;
            this.__account_path = `m/84'/0'/${this.account}'`;
          }
          if ((this.pathType !== null) && (this.pathType !== "custom"))
            from = window.BIP32_XKeyToPathXKey(from, this.pathType);
          fromType = "xPriv";
        }
      } else throw new Error("invalid initial data");


      let rawFrom = window.decodeBase58(from, { checkSum: true, hex: false });
      this.testnet = window.xKeyNetworkType(rawFrom) === 'testnet';
      this.version = rawFrom.slice(0, 4).hex();
      this.depth = rawFrom[4];

      if (this.pathType !== "custom") {
        if ((this.depth === 0) || (this.depth === 3)) {
          let l = this.path.split('/');
          this.__path = l.slice(this.depth, 4).join('/');
        } else {
          this.pathType = 'custom';
          this.path = "m";
        }
      }

      this.fingerprint = rawFrom.slice(5, 9).hex();
      this.child = rawFrom.readUIntBE(9, 4);
      this.chainCode = rawFrom.slice(9, 4).hex();
      if (fromType === "xPriv") {
        if (this.depth === 0) this.masterXPrivateKey = from;

        if (this.pathType !== "custom") {

          /**
          * account private xkey (string)
          */
          this.accountXPrivateKey = window.deriveXKey(from, this.__path, { subPath: true });

          /**
          * account public xkey (string)
          */
          this.accountXPublicKey = window.xPrivateToXPublicKey(this.accountXPrivateKey);

          this.accountXPrivateKey = window.deriveXKey(from, this.__path, { subPath: true });
          this.accountXPublicKey = window.xPrivateToXPublicKey(this.accountXPrivateKey);

          this.externalChainXPrivateKey = window.deriveXKey(from, this.__path + `/${this.chain}`, { subPath: true });
          this.externalChainXPublicKey = window.xPrivateToXPublicKey(this.externalChainXPrivateKey);

          this.internalChainXPrivateKey = window.deriveXKey(from, this.__path + `/${this.chain + 1}`, { subPath: true });
          this.internalChainXPublicKey = window.xPrivateToXPublicKey(this.internalChainXPrivateKey);
        } else {
          this.chainXPrivateKey = window.deriveXKey(from, this.path);
          this.chainXPublicKey = window.xPrivateToXPublicKey(this.chainXPrivateKey);
        }


      } else {
        if (this.pathType !== "custom") {
          this.accountXPublicKey = from;
          this.externalChainXPublicKey = window.deriveXKey(from, this.__path + `/${this.chain}`, { subPath: true });
          this.internalChainXPrivateKey = window.deriveXKey(from, this.__path + `/${this.chain + 1}`, { subPath: true });
        } else {
          this.chainXPublicKey = window.deriveXKey(from, this.path);
        }

      }
      if (this.mnemonic !== null) {
        this.sharesThreshold = A.threshold;
        this.sharesTotal = A.shares;
        if (this.sharesThreshold > this.sharesTotal) throw new Error("Threshold invalid");
        if (this.sharesTotal > 1) {
          let m = this.mnemonic.trim().split(/\s+/);
          let bitSize = m.length * 11;
          let checkSumBitLen = bitSize % 32;
          if (this.sharesTotal > (2 ** checkSumBitLen - 1))
            throw new Error(`Maximum ${2 ** checkSumBitLen - 1} shares allowed for ${m.length} mnemonic words`);
          this.mnemonicShares = window.splitMnemonic(A.threshold, A.shares, this.mnemonic,
            { wordList: A.BIP39_WORDLIST, embeddedIndex: true });

        }
      }

      if (A.addressType !== null) this.addressType = A.addressType;
      else {
        if (this.pathType === "BIP84") this.addressType = "P2WPKH";
        else if (this.pathType === "BIP49") this.addressType = "P2SH_P2WPKH";
        else this.addressType = "P2PKH";
      }
    }


  }

  Wallet.prototype.setChain = function (i) {
    this.chain = i;
  };

  /**
  * the class method for creating a wallet address.
  *
  * :parameters:
  *   :index
  *   :external: (optional) boolean, by default is ``true``
  * :return: object:
  *
  *    - address
  *    - publicKey
  *    - privateKey (in case wallet is restored from private xkey or mnemonic)
  */
  Wallet.prototype.getAddress = function (index, external = true) {
    let r = {};
    let h = (this.hardenedAddresses) ? "'" : "";
    if (this.pathType !== 'custom') {
      let p = "m/" + index + h;
      r.path = `${this.__account_path}/${this.chain + !external}/${index}${h}`;
      if (external) {
        if (this.externalChainXPrivateKey !== undefined) {
          let key = window.deriveXKey(this.externalChainXPrivateKey, p);
          r.privateKey = window.privateFromXPrivateKey(key);
          r.publicKey = window.privateToPublicKey(r.privateKey);
        } else {
          let key = window.deriveXKey(this.externalChainXPublicKey, p);

          r.publicKey = window.publicFromXPublicKey(key);
        }
      } else {
        if (this.internalChainXPrivateKey !== undefined) {
          let key = window.deriveXKey(this.internalChainXPrivateKey, p);
          r.privateKey = window.privateFromXPrivateKey(key);
          r.publicKey = window.privateToPublicKey(r.privateKey);
        } else {
          let key = window.deriveXKey(this.internalChainXPublicKey, p);
          r.publicKey = window.publicFromXPublicKey(key);
        }

      }
    } else {

      let p = "m/" + index + h;
      r.path = this.path + "/" + index + h;
      if (this.chainXPrivateKey !== undefined) {
        let key = window.deriveXKey(this.chainXPrivateKey, p);
        r.privateKey = window.privateFromXPrivateKey(key);
        r.publicKey = window.privateToPublicKey(r.privateKey);
      } else {
        let key = window.deriveXKey(this.chainXPublicKey, p);

        r.publicKey = window.publicFromXPublicKey(key);
      }
    }

    if (this.addressType === "P2WPKH") r.address = window.publicKeyToAddress(r.publicKey, { testnet: this.testnet });
    else if (this.addressType === "P2SH_P2WPKH") r.address = window.publicKeyToAddress(r.publicKey,
      { p2sh_p2wpkh: true, testnet: this.testnet });
    else if (this.addressType === "P2PKH") r.address = window.publicKeyToAddress(r.publicKey,
      { witnessVersion: null, testnet: this.testnet });
    return r;
  };

  window.Wallet = Wallet;

}

function constants(window) {
  let BF = window.Buffer.from;
  let INT_BASE32_MAP = {};
  let BASE32_INT_MAP = {};
  let BASE32CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
  let BASE32CHARSET_UPCASE = "QPZRY9X8GF2TVDW0S3JN54KHCE6MUA7L";

  for (let i = 0; i < BASE32CHARSET.length; i++) {
    INT_BASE32_MAP[BASE32CHARSET[i]] = i;
    BASE32_INT_MAP[i] = BASE32CHARSET.charCodeAt(i);
  }
  for (let i = 0; i < BASE32CHARSET_UPCASE.length; i++) INT_BASE32_MAP[BASE32CHARSET_UPCASE[i]] = i;

  const MAINNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX = '5';
  const MAINNET_PRIVATE_KEY_COMPRESSED_PREFIX = 'K';
  const MAINNET_PRIVATE_KEY_COMPRESSED_PREFIX_2 = 'L';
  const TESTNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX = '9';
  const TESTNET_PRIVATE_KEY_COMPRESSED_PREFIX = 'c';
  const MAINNET_ADDRESS_PREFIX = '1';
  const TESTNET_ADDRESS_PREFIX = 'm';
  const TESTNET_ADDRESS_PREFIX_2 = 'n';
  const MAINNET_SCRIPT_ADDRESS_PREFIX = '3';
  const TESTNET_SCRIPT_ADDRESS_PREFIX = '2';

  window.BASE32CHARSET = BASE32CHARSET;
  window.BASE32CHARSET_UPCASE = BASE32CHARSET_UPCASE;
  window.INT_BASE32_MAP = INT_BASE32_MAP;
  window.BASE32_INT_MAP = BASE32_INT_MAP;
  window.MAINNET_ADDRESS_PREFIX = MAINNET_ADDRESS_PREFIX;
  window.TESTNET_ADDRESS_PREFIX = TESTNET_ADDRESS_PREFIX;
  window.TESTNET_ADDRESS_PREFIX_2 = TESTNET_ADDRESS_PREFIX_2;
  window.MAINNET_SCRIPT_ADDRESS_PREFIX = MAINNET_SCRIPT_ADDRESS_PREFIX;
  window.TESTNET_SCRIPT_ADDRESS_PREFIX = TESTNET_SCRIPT_ADDRESS_PREFIX;
  window.MAINNET_SEGWIT_ADDRESS_PREFIX = 'bgl';
  window.TESTNET_SEGWIT_ADDRESS_PREFIX = 'tbgl';
  window.MAINNET_ADDRESS_BYTE_PREFIX = [0];
  window.TESTNET_ADDRESS_BYTE_PREFIX = [111];
  window.MAINNET_SCRIPT_ADDRESS_BYTE_PREFIX = [5];
  window.TESTNET_SCRIPT_ADDRESS_BYTE_PREFIX = [196];
  window.MAINNET_SEGWIT_ADDRESS_BYTE_PREFIX = [3, 3, 3, 0, 2, 7, 12];
  window.TESTNET_SEGWIT_ADDRESS_BYTE_PREFIX = [3, 3, 0, 20, 2];
  window.TESTNET_PRIVATE_KEY_BYTE_PREFIX = [0xef];
  window.MAINNET_PRIVATE_KEY_BYTE_PREFIX = [0x80];
  window.MAINNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX = MAINNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX;
  window.MAINNET_PRIVATE_KEY_COMPRESSED_PREFIX = MAINNET_PRIVATE_KEY_COMPRESSED_PREFIX;
  window.MAINNET_PRIVATE_KEY_COMPRESSED_PREFIX_2 = MAINNET_PRIVATE_KEY_COMPRESSED_PREFIX_2;
  window.TESTNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX = TESTNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX;
  window.TESTNET_PRIVATE_KEY_COMPRESSED_PREFIX = TESTNET_PRIVATE_KEY_COMPRESSED_PREFIX;
  window.ADDRESS_PREFIX_LIST = [MAINNET_ADDRESS_PREFIX,
    TESTNET_ADDRESS_PREFIX,
    TESTNET_ADDRESS_PREFIX_2,
    MAINNET_SCRIPT_ADDRESS_PREFIX,
    TESTNET_SCRIPT_ADDRESS_PREFIX];
  window.PRIVATE_KEY_PREFIX_LIST = [MAINNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX,
    MAINNET_PRIVATE_KEY_COMPRESSED_PREFIX,
    MAINNET_PRIVATE_KEY_COMPRESSED_PREFIX_2,
    TESTNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX,
    TESTNET_PRIVATE_KEY_COMPRESSED_PREFIX];
  window.SCRIPT_TYPES = {
    "P2PKH": 0,
    "P2SH": 1,
    "PUBKEY": 2,
    "NULL_DATA": 3,
    "MULTISIG": 4,
    "P2WPKH": 5,
    "P2WSH": 6,
    "NON_STANDARD": 7,
    "NULL_DATA_NON_STANDARD": 8
  };
  window.SCRIPT_N_TYPES = {
    0: "P2PKH",
    1: "P2SH",
    2: "PUBKEY",
    3: "NULL_DATA",
    4: "MULTISIG",
    5: "P2WPKH",
    6: "P2WSH",
    7: "NON_STANDARD",
    8: "NULL_DATA_NON_STANDARD"
  };
  window.GAMMA_NUM_LN = 607 / 128;
  window.GAMMA_TABLE_LN = [0.99999999999999709182,
    57.156235665862923517,
    -59.597960355475491248,
    14.136097974741747174,
    -0.49191381609762019978,
    0.33994649984811888699e-4,
    0.46523628927048575665e-4,
    -0.98374475304879564677e-4,
    0.15808870322491248884e-3,
    -0.21026444172410488319e-3,
    0.21743961811521264320e-3,
    -0.16431810653676389022e-3,
    0.84418223983852743293e-4,
    -0.26190838401581408670e-4,
    0.36899182659531622704e-5];
  window.MACHEP = 1.11022302462515654042E-16;
  window.MAXLOG = 7.09782712893383996732E2;
  window.ECDSA_SEC256K1_ORDER = new window.BN('fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141', 16);
  window.SECP256K1_CONTEXT_VERIFY = (1 << 0) | (1 << 8);
  window.SECP256K1_CONTEXT_SIGN = (1 << 0) | (1 << 9);
  window.SECP256K1_CONTEXT_NONE = (1 << 0);
  window.SECP256K1_EC_COMPRESSED = (1 << 1) | (1 << 8);
  window.SECP256K1_EC_UNCOMPRESSED = (1 << 1);
  window.MAX_AMOUNT = 2100000000000000;
  window.SIGHASH_ALL = 0x00000001;
  window.SIGHASH_NONE = 0x00000002;
  window.SIGHASH_SINGLE = 0x00000003;
  window.SIGHASH_ANYONECANPAY = 0x00000080;
  window.HARDENED_KEY = 0x80000000;
  window.FIRST_HARDENED_CHILD = 0x80000000;
  window.PATH_LEVEL_BIP0044 = [0x8000002C, 0x80000000, 0x80000000, 0, 0];
  window.TESTNET_PATH_LEVEL_BIP0044 = [0x8000002C, 0x80000001, 0x80000000, 0, 0];
  // CONSTANTS hierarchical deterministic wallets (HD Wallets)
  // m/44'/0' P2PKH
  window.MAINNET_XPRIVATE_KEY_PREFIX = BF([0x04, 0x88, 0xad, 0xe4]);
  window.MAINNET_XPUBLIC_KEY_PREFIX = BF([0x04, 0x88, 0xb2, 0x1e]);
  // m/44'/1' P2PKH
  window.TESTNET_XPRIVATE_KEY_PREFIX = BF([0x04, 0x35, 0x83, 0x94]);
  window.TESTNET_XPUBLIC_KEY_PREFIX = BF([0x04, 0x35, 0x87, 0xcf]);
  // m/44'/0' P2PKH
  window.MAINNET_M44_XPRIVATE_KEY_PREFIX = BF([0x04, 0x88, 0xad, 0xe4]);
  window.MAINNET_M44_XPUBLIC_KEY_PREFIX = BF([0x04, 0x88, 0xb2, 0x1e]);
  // m/44'/1' P2PKH
  window.TESTNET_M44_XPRIVATE_KEY_PREFIX = BF([0x04, 0x35, 0x83, 0x94]);
  window.TESTNET_M44_XPUBLIC_KEY_PREFIX = BF([0x04, 0x35, 0x87, 0xcf]);
  // m/49'/0' P2WPKH in P2SH
  window.MAINNET_M49_XPRIVATE_KEY_PREFIX = BF([0x04, 0x9d, 0x78, 0x78]);
  window.MAINNET_M49_XPUBLIC_KEY_PREFIX = BF([0x04, 0x9d, 0x7c, 0xb2]);
  // m/49'/1' P2WPKH in P2SH
  window.TESTNET_M49_XPRIVATE_KEY_PREFIX = BF([0x04, 0x4a, 0x4e, 0x28]);
  window.TESTNET_M49_XPUBLIC_KEY_PREFIX = BF([0x04, 0x4a, 0x52, 0x62]);
  // m/84'/0' P2WPKH
  window.MAINNET_M84_XPRIVATE_KEY_PREFIX = BF([0x04, 0xb2, 0x43, 0x0c]);
  window.MAINNET_M84_XPUBLIC_KEY_PREFIX = BF([0x04, 0xb2, 0x47, 0x46]);
  // m/84'/1' P2WPKH
  window.TESTNET_M84_XPRIVATE_KEY_PREFIX = BF([0x04, 0x5f, 0x18, 0xbc]);
  window.TESTNET_M84_XPUBLIC_KEY_PREFIX = BF([0x04, 0x5f, 0x1c, 0xf6]);
}

function address(window) {
  let Buffer = window.Buffer;
  let ARGS = window.defArgs;
  let getBuffer = window.getBuffer;
  let BF = Buffer.from;
  let BC = Buffer.concat;
  let O = window.OPCODE;



  /**
  * Get address from public key/script hash. In case PUBKEY, P2PKH, P2PKH public key/script hash is SHA256+RIPEMD160,P2WSH script hash is SHA256.
  *
  * :parameters:
  *   :ha: public key hash or script hash in HEX or bytes string format.
  * :param testnet: (optional) flag for testnet network, by default is ``false``.
  * :param scriptHash: (optional) flag for script hash (P2SH address), by default is ``false``.
  * :param witnessVersion: (optional) witness program version, by default is 0, for legacy  address format use null.
  * :return: address in base58 or bech32 format.
  */
  window.hashToAddress = (ha, A = {}) => {
    ARGS(A, { testnet: false, scriptHash: false, witnessVersion: 0 });
    ha = getBuffer(ha);
    let prefix;
    if (!A.scriptHash) {
      if (A.witnessVersion === null) {
        if (ha.length !== 20) throw new Error('address hash length incorrect');
        if (A.testnet) prefix = BF(window.TESTNET_ADDRESS_BYTE_PREFIX);
        else prefix = BF(window.MAINNET_ADDRESS_BYTE_PREFIX);
        let h = BC([prefix, ha]);
        h = BC([h, window.doubleSha256(h, { hex: false }).slice(0, 4)]);
        return window.encodeBase58(h);
      } else if ((ha.length !== 20) && (ha.length !== 32))
        throw new Error('address hash length incorrect');
    }

    if (A.witnessVersion === null) {
      if (A.testnet) prefix = BF(window.TESTNET_SCRIPT_ADDRESS_BYTE_PREFIX);
      else prefix = BF(window.MAINNET_SCRIPT_ADDRESS_BYTE_PREFIX);
      let h = BC([prefix, ha]);
      h = BC([h, window.doubleSha256(h, { hex: false }).slice(0, 4)]);
      return window.encodeBase58(h);
    }

    let hrp;
    if (A.testnet) {
      prefix = window.TESTNET_SEGWIT_ADDRESS_BYTE_PREFIX;
      hrp = window.TESTNET_SEGWIT_ADDRESS_PREFIX;
    } else {
      prefix = window.MAINNET_SEGWIT_ADDRESS_BYTE_PREFIX;
      hrp = window.MAINNET_SEGWIT_ADDRESS_PREFIX;
    }
    ha = window.rebase_8_to_5(Array.from(ha));
    ha.unshift(A.witnessVersion);

    let checksum = window.bech32Polymod(prefix.concat(ha.concat([0, 0, 0, 0, 0, 0])));
    checksum = window.rebase_8_to_5(window.intToBytes(checksum, 5, 'big')).slice(2);
    return hrp + '1' + window.rebase_5_to_32(ha.concat(checksum), false);
  };

  /**
  * Get address hash from base58 or bech32 address format.
  *
  * :parameters:
  *   :a: address in base58 or bech32 format.
  * :param hex:  (optional) If set to ``true`` return key in HEX format, by default is ``false``.
  * :return: script in HEX or bytes string.
  */
  window.addressToHash = (a, A = {}) => {
    ARGS(A, { hex: false });
    if (!window.isString(a)) throw new Error('address invalid');
    let h;
    if (window.ADDRESS_PREFIX_LIST.includes(a[0])) {
      h = window.decodeBase58(a, { hex: false });
      h = h.slice(1, h.length - 4);
    } else if ([window.MAINNET_SEGWIT_ADDRESS_PREFIX,
    window.TESTNET_SEGWIT_ADDRESS_PREFIX].includes(a.split('1')[0])) {
      let q = window.rebase_32_to_5(a.split('1')[1]);
      h = BF(window.rebase_5_to_8(q.slice(1, q.length - 6), false));
    } else return null;
    return (A.hex) ? h.hex() : h;
  };

  /**
  * Get address from public key/script hash. In case PUBKEY, P2PKH, P2PKH public key/script hash is SHA256+RIPEMD160, P2WSH script hash is SHA256.
  *
  * :parameters:
  *   :pubkey: public key HEX or bytes string format.
  * :param testnet: (optional) flag for testnet network, by default is ``false``.
  * :param p2sh_p2wpkh: (optional) flag for P2WPKH inside P2SH address, by default is ``false``.
  * :param witnessVersion: (optional) witness program version, by default is 0, for legacy address format use null.
  * :return: address in base58 or bech32 format.
  */
  window.publicKeyToAddress = (pubkey, A = {}) => {
    ARGS(A, { testnet: false, p2sh_p2wpkh: false, witnessVersion: 0 });
    pubkey = getBuffer(pubkey);
    let h;
    if (A.p2sh_p2wpkh) {
      if (pubkey.length !== 33) throw new Error('public key length invalid');
      h = window.hash160(BC([BF([0, 20]), window.hash160(pubkey)]));
      A.witnessVersion = null;
    } else {
      if (A.witnessVersion !== null)
        if (pubkey.length !== 33) throw new Error('public key length invalid');
      h = window.hash160(pubkey);
    }
    A.scriptHash = A.p2sh_p2wpkh;
    return window.hashToAddress(h, A);
  };

  /**
  * Get address type.
  *
  * :parameters:
  *   :a: address in base58 or bech32 format.
  * :param num: (optional) If set to ``true`` return type in numeric format, by default is ``false``.
  * :return: address type in string or numeric format.
  */
  window.addressType = (a, A = {}) => {
    ARGS(A, { num: false });
    if ([window.TESTNET_SCRIPT_ADDRESS_PREFIX, window.MAINNET_SCRIPT_ADDRESS_PREFIX].includes(a[0]))
      return (A.num) ? window.SCRIPT_TYPES["P2SH"] : "P2SH";
    if ([window.MAINNET_ADDRESS_PREFIX, window.TESTNET_ADDRESS_PREFIX, window.TESTNET_ADDRESS_PREFIX_2].includes(a[0]))
      return (A.num) ? window.SCRIPT_TYPES["P2PKH"] : "P2PKH";

    if (window.MAINNET_SEGWIT_ADDRESS_PREFIX === a.slice(0, 3)) {
      if (a.length === 43) return (A.num) ? window.SCRIPT_TYPES["P2WPKH"] : "P2WPKH";
      if (a.length === 63) return (A.num) ? window.SCRIPT_TYPES["P2WSH"] : "P2WSH";
    }
    if (window.TESTNET_SEGWIT_ADDRESS_PREFIX === a.slice(0, 4)) {
      if (a.length === 44) return (A.num) ? window.SCRIPT_TYPES["P2WPKH"] : "P2WPKH";
      if (a.length === 64) return (A.num) ? window.SCRIPT_TYPES["P2WSH"] : "P2WSH";
    }
    return (A.num) ? window.SCRIPT_TYPES["NON_STANDARD"] : "NON_STANDARD";
  };

  window.addressNetType = (a) => {
    if ([window.MAINNET_SCRIPT_ADDRESS_PREFIX, window.MAINNET_ADDRESS_PREFIX].includes(a[0])) return "mainnet";
    if (a.slice(0, 3) === window.MAINNET_SEGWIT_ADDRESS_PREFIX) return "mainnet";
    if ([window.TESTNET_SCRIPT_ADDRESS_PREFIX,
    window.TESTNET_ADDRESS_PREFIX, window.TESTNET_ADDRESS_PREFIX_2].includes(a[0])) return "testnet";
    if (a.slice(0, 4) === window.TESTNET_SEGWIT_ADDRESS_PREFIX) return "testnet";
    return null;
  };

  /**
  * Get public key script from address.
  *
  * :parameters:
  *   :a: address in base58 or bech32 format.
  * :param hex:  (optional) If set to ``true`` return key in HEX format, by default is ``false``.
  * :return: public key script in HEX or bytes string.
  */
  window.addressToScript = (a, A = {}) => {
    ARGS(A, { hex: false });
    if (!window.isString(a)) throw new Error('address invalid');
    let s;
    if ([window.TESTNET_SCRIPT_ADDRESS_PREFIX, window.MAINNET_SCRIPT_ADDRESS_PREFIX].includes(a[0])) {
      s = BC([BF([O.OP_HASH160, 0x14]), window.addressToHash(a), BF([O.OP_EQUAL])]);
      return (A.hex) ? s.hex() : s;
    }
    if ([window.MAINNET_ADDRESS_PREFIX, window.TESTNET_ADDRESS_PREFIX, window.TESTNET_ADDRESS_PREFIX_2].includes(a[0])) {
      s = BC([BF([O.OP_DUP, O.OP_HASH160, 0x14]), window.addressToHash(a), BF([O.OP_EQUALVERIFY, O.OP_CHECKSIG])]);
      return (A.hex) ? s.hex() : s;
    }
    if ([window.TESTNET_SEGWIT_ADDRESS_PREFIX, window.MAINNET_SEGWIT_ADDRESS_PREFIX].includes(a.split("1")[0])) {
      let h = window.addressToHash(a);
      s = BC([BF([O.OP_0, h.length]), window.addressToHash(a)]);
      return (A.hex) ? s.hex() : s;
    }
    throw new Error('address invalid');
  };

  window.getWitnessVersion = (address) => window.rebase_32_to_5(address.split(1)[1])[0];


  /**
  * Check is address valid.
  *
  * :parameters:
  *   :address: address in base58 or bech32 format.
  * :param testnet: (optional) flag for testnet network, by default is ``false``.
  * :return: boolean.
  */
  window.isAddressValid = (address, A = {}) => {
    ARGS(A, { testnet: false });
    if (!window.isString(address)) return false;

    if ([window.MAINNET_ADDRESS_PREFIX,
    window.MAINNET_SCRIPT_ADDRESS_PREFIX,
    window.TESTNET_ADDRESS_PREFIX,
    window.TESTNET_ADDRESS_PREFIX_2,
    window.TESTNET_SCRIPT_ADDRESS_PREFIX].includes(address[0])) {
      if (A.testnet === true) {
        if (!([window.TESTNET_ADDRESS_PREFIX,
        window.TESTNET_ADDRESS_PREFIX_2,
        window.TESTNET_SCRIPT_ADDRESS_PREFIX].includes(address[0]))) return false;
      } else if (![window.MAINNET_ADDRESS_PREFIX,
      window.MAINNET_SCRIPT_ADDRESS_PREFIX].includes(address[0])) return false;
      let b = window.decodeBase58(address, { hex: false });
      if (b.length !== 25) return false;
      let checksum = b.slice(-4);
      let verifyChecksum = window.doubleSha256(b.slice(0, -4)).slice(0, 4);
      return checksum.equals(verifyChecksum);

    } else {
      let prefix, payload;
      if ([window.TESTNET_SEGWIT_ADDRESS_PREFIX,
      window.MAINNET_SEGWIT_ADDRESS_PREFIX].includes(address.split("1")[0].toLowerCase())) {
        if (address.length !== 43 && address.length !== 63 && address.length !== 44 && address.length !== 64) return false;
        let pp = address.split('1');
        prefix = pp[0];
        payload = pp[1];
        let upp;
        upp = prefix[0] !== prefix[0].toLowerCase();
        for (let i = 0; i < payload.length; i++)
          if (upp === true) {
            if (window.BASE32CHARSET_UPCASE.indexOf(payload[i]) === -1) return false;
          } else {
            if (window.BASE32CHARSET.indexOf(payload[i]) === -1) return false;
          }
        payload = payload.toLowerCase();
        prefix = prefix.toLowerCase();
        let strippedPrefix;
        if (A.testnet === true) {
          if (prefix !== window.TESTNET_SEGWIT_ADDRESS_PREFIX) return false;
          strippedPrefix = window.TESTNET_SEGWIT_ADDRESS_BYTE_PREFIX;
        } else {
          if (prefix !== window.MAINNET_SEGWIT_ADDRESS_PREFIX) return false;
          strippedPrefix = window.MAINNET_SEGWIT_ADDRESS_BYTE_PREFIX;
        }
        let d = window.rebase_32_to_5(payload);
        let h = d.slice(0, -6);
        let checksum = d.slice(-6);
        strippedPrefix = strippedPrefix.concat(h).concat([0, 0, 0, 0, 0, 0]);
        let checksum2 = window.bech32Polymod(strippedPrefix);
        checksum2 = window.rebase_8_to_5(window.intToBytes(checksum2, 5, 'big')).slice(2);
        return window.bytesToString(checksum) === window.bytesToString(checksum2);
      }
      return false;
    }
  };
}

function bip32(window) {
  let CM = window.__bitcoin_core_crypto.module;
  CM._malloc;
  CM._free;
  let BA = window.Buffer.alloc;
  let BC = window.Buffer.concat;
  let BF = window.Buffer.from;
  let ARGS = window.defArgs;
  window.getBuffer;
  let BN = window.BN;
  CM.getValue;

  /**
  * Create extended private key from seed
  *
  * :parameters:
  *   :seed: seed HEX or bytes string.
  * :param testnet: (optional) flag for testnet network, by default is ``false``.
  * :param base58: (optional) return result as base58 encoded string, by default is ``true``.
  * :return: extended private key  in base58 string format.
  */
  window.createMasterXPrivateKey = (seed, A = {}) => {
    ARGS(A, { testnet: false, base58: true });
    let i = window.hmacSha512("Bitcoin seed", seed);
    let m = i.slice(0, 32);
    let c = i.slice(32);
    let mi = new BN(m);
    if ((mi.gte(window.ECDSA_SEC256K1_ORDER) || mi.lte(1))) return null;
    let key = (A.testnet) ? window.TESTNET_XPRIVATE_KEY_PREFIX : window.MAINNET_XPRIVATE_KEY_PREFIX;
    key = BC([key, BA(9, 0), c, BA(1, 0), m]);
    if (A.base58) return window.encodeBase58(BC([key, window.doubleSha256(key).slice(0, 4)]));
    return key;
  };

  /**
  * Get extended public key from extended private key using ECDSA secp256k1
  *
  * :parameters:
  *   :xKey: extended private key in base58, HEX or bytes string.
  * :param base58: (optional) return result as base58 encoded string, by default is ``true``.
  * :return: extended private key  in base58 string format.
  */
  window.xPrivateToXPublicKey = (xKey, A = {}) => {
    ARGS(A, { base58: true });
    if (window.isString(xKey)) xKey = window.decodeBase58(xKey, { hex: false }).slice(0, -4);
    if (!window.isBuffer(xKey)) throw new Error("invalid xPrivateKey");
    if (xKey.length !== 78) throw new Error("invalid xPrivateKey");
    let prefix;
    if (xKey.slice(0, 4).equals(window.TESTNET_XPRIVATE_KEY_PREFIX)) prefix = window.TESTNET_XPUBLIC_KEY_PREFIX;
    else if (xKey.slice(0, 4).equals(window.MAINNET_XPRIVATE_KEY_PREFIX)) prefix = window.MAINNET_XPUBLIC_KEY_PREFIX;

    else if (xKey.slice(0, 4).equals(window.MAINNET_M49_XPRIVATE_KEY_PREFIX)) prefix = window.MAINNET_M49_XPUBLIC_KEY_PREFIX;
    else if (xKey.slice(0, 4).equals(window.TESTNET_M49_XPRIVATE_KEY_PREFIX)) prefix = window.TESTNET_M49_XPUBLIC_KEY_PREFIX;

    else if (xKey.slice(0, 4).equals(window.MAINNET_M84_XPRIVATE_KEY_PREFIX)) prefix = window.MAINNET_M84_XPUBLIC_KEY_PREFIX;
    else if (xKey.slice(0, 4).equals(window.TESTNET_M84_XPRIVATE_KEY_PREFIX)) prefix = window.TESTNET_M84_XPUBLIC_KEY_PREFIX;

    else throw new Error("invalid xPrivateKey");
    let key = BC([prefix, xKey.slice(4, 45),
      window.privateToPublicKey(xKey.slice(46), { hex: false })]);
    if (A.base58) return window.encodeBase58(BC([key, window.doubleSha256(key).slice(0, 4)]));
    return key;
  };

  window.__decodePath = (p, subPath = false) => {
    p = p.split('/');
    if (!subPath)
      if (p[0] !== 'm') throw new Error("invalid path");
    let r = [];
    for (let i = 1; i < p.length; i++) {
      let k = parseInt(p[i]);
      if ((p[i][p[i].length - 1] === "'") && (k < window.HARDENED_KEY)) k += window.HARDENED_KEY;
      r.push(k);
    }
    return r;
  };

  /**
  * Child Key derivation for extended private/public keys
  *
  * :parameters:
  *   :xKey: extended private/public in base58, HEX or bytes string format.
  *   :path: list of derivation path levels. For hardened derivation use HARDENED_KEY flag.
  * :param base58: (optional) return result as base58 encoded string, by default is ``true``.
  * :param subPath: (optional) boolean, by default is ``false``.
  * :return: extended child private/public key  in base58, HEX or bytes string format.
  */
  window.deriveXKey = (xKey, path, A = {}) => {
    ARGS(A, { base58: true, subPath: false });
    if (window.isString(xKey)) xKey = window.decodeBase58(xKey, { checkSum: true, hex: false });
    path = window.__decodePath(path, A.subPath);

    if (window.xKeyType(xKey) === "private") for (let p of path) xKey = window.__deriveChildXPrivateKey(xKey, p);
    else for (let p of path) xKey = window.__deriveChildXPublicKey(xKey, p);

    if (A.base58) return window.encodeBase58(xKey, { checkSum: true });
    return xKey;
  };


  window.__deriveChildXPrivateKey = (xPrivateKey, i) => {
    let c = xPrivateKey.slice(13, 45);
    let k = xPrivateKey.slice(45);
    let depth = xPrivateKey[4] + 1;
    if (depth > 255) throw new Error("path depth should be <= 255");
    let r = BF(k.slice(1));
    let pub = window.privateToPublicKey(r, { hex: false });
    let fingerprint = window.hash160(pub).slice(0, 4);
    let s = window.hmacSha512(c, BC([(i >= window.HARDENED_KEY) ? k : pub, BF(window.intToBytes(i, 4, "big"))]));
    let pi = new BN(s.slice(0, 32));
    if ((pi.gte(window.ECDSA_SEC256K1_ORDER))) return null;
    let ki = new BN(k.slice(1));
    ki = ki.add(pi);
    ki = ki.mod(window.ECDSA_SEC256K1_ORDER);
    if (ki.isZero()) return null;
    let key = ki.toArrayLike(window.Buffer, 'be', 32);
    return BC([xPrivateKey.slice(0, 4), BF([depth]), fingerprint, BF(window.intToBytes(i, 4, "big")),
    s.slice(32), BA(1, 0), key]);
  };

  window.__deriveChildXPublicKey = (xPublicKey, i) => {
    let c = xPublicKey.slice(13, 45);
    let k = xPublicKey.slice(45);
    let depth = xPublicKey[4] + 1;
    if (depth > 255) throw new Error("path depth should be <= 255");
    if (i >= window.HARDENED_KEY) throw new Error("derivation from extended public key impossible");
    let fingerprint = window.hash160(k).slice(0, 4);
    let s = window.hmacSha512(c, BC([k, BF(window.intToBytes(i, 4, "big"))]));
    let pi = new BN(s.slice(0, 32));
    if ((pi.gte(window.ECDSA_SEC256K1_ORDER))) return null;
    let pk = window.publicKeyAdd(k, s.slice(0, 32), { hex: false });
    return BC([xPublicKey.slice(0, 4), BF([depth]), fingerprint, BF(window.intToBytes(i, 4, "big")), s.slice(32), pk]);
  };

  /**
  * Get public key from extended public key
  *
  * :parameters:
  *   :xPub: extended public in base58, HEX or bytes string format.
  * :param hex: (optional) return result as HEX encoded string, by default is ``true``.
  * :return: public key  in HEX or bytes string format.
  */
  window.publicFromXPublicKey = (xPub, A = {}) => {
    ARGS(A, { hex: true });
    if (window.isString(xPub)) xPub = window.decodeBase58(xPub, { checkSum: true, hex: false });
    if (xPub.length !== 78) throw new Error("invalid extended public key");
    return (A.hex) ? xPub.slice(45).hex() : xPub.slice(45)
  };

  /**
  * Get private key from extended private key
  *
  * :parameters:
  *   :xPriv: extended private in base58, HEX or bytes string format.
  * :param wif: (optional) return result as WIF format, by default is ``true``.
  * :return: private key  in HEX or bytes string format.
  */
  window.privateFromXPrivateKey = (xPriv, A = {}) => {
    ARGS(A, { wif: true });
    if (window.isString(xPriv)) xPriv = window.decodeBase58(xPriv, { checkSum: true, hex: false });
    if (xPriv.length !== 78) throw new Error("invalid extended public key");
    let prefix = xPriv.slice(0, 4);
    let testnet;
    if (prefix.equals(window.MAINNET_XPRIVATE_KEY_PREFIX)) testnet = false;
    else if (prefix.equals(window.TESTNET_XPRIVATE_KEY_PREFIX)) testnet = true;
    else if (prefix.equals(window.MAINNET_M49_XPRIVATE_KEY_PREFIX)) testnet = false;
    else if (prefix.equals(window.TESTNET_M49_XPRIVATE_KEY_PREFIX)) testnet = true;
    else if (prefix.equals(window.MAINNET_M84_XPRIVATE_KEY_PREFIX)) testnet = false;
    else if (prefix.equals(window.TESTNET_M84_XPRIVATE_KEY_PREFIX)) testnet = true;
    else
      throw new Error("invalid extended public key");
    return (A.wif) ? window.privateKeyToWif(xPriv.slice(46), { testnet: testnet, wif: true }) : xPriv.slice(46)
  };

  window.isXPrivateKeyValid = (xPriv) => {
    if (window.isString(xPriv)) xPriv = window.decodeBase58(xPriv, { checkSum: true, hex: false });
    if (xPriv.length !== 78) return false;
    let prefix = xPriv.slice(0, 4);
    if (prefix.equals(window.MAINNET_XPRIVATE_KEY_PREFIX)) return true;
    if (prefix.equals(window.TESTNET_XPRIVATE_KEY_PREFIX)) return true;
    if (prefix.equals(window.MAINNET_M49_XPRIVATE_KEY_PREFIX)) return true;
    if (prefix.equals(window.TESTNET_M49_XPRIVATE_KEY_PREFIX)) return true;
    if (prefix.equals(window.MAINNET_M84_XPRIVATE_KEY_PREFIX)) return true;
    return prefix.equals(window.TESTNET_M84_XPRIVATE_KEY_PREFIX);
  };

  window.isXPublicKeyValid = (xPub) => {
    if (window.isString(xPub)) xPub = window.decodeBase58(xPub, { checkSum: true, hex: false });
    if (xPub.length !== 78) return false;
    let prefix = xPub.slice(0, 4);
    if (prefix.equals(window.MAINNET_XPUBLIC_KEY_PREFIX)) return true;
    if (prefix.equals(window.TESTNET_XPUBLIC_KEY_PREFIX)) return true;
    if (prefix.equals(window.MAINNET_M49_XPUBLIC_KEY_PREFIX)) return true;
    if (prefix.equals(window.TESTNET_M49_XPUBLIC_KEY_PREFIX)) return true;
    if (prefix.equals(window.MAINNET_M84_XPUBLIC_KEY_PREFIX)) return true;
    return prefix.equals(window.TESTNET_M84_XPUBLIC_KEY_PREFIX);
  };

  window.xKeyNetworkType = (xKey) => {
    if (window.isString(xKey)) xKey = window.decodeBase58(xKey, { checkSum: true, hex: false });
    if (xKey.length !== 78) return false;
    let prefix = xKey.slice(0, 4);
    if (prefix.equals(window.MAINNET_XPRIVATE_KEY_PREFIX)) return "mainnet";
    if (prefix.equals(window.TESTNET_XPRIVATE_KEY_PREFIX)) return "testnet";
    if (prefix.equals(window.MAINNET_M49_XPRIVATE_KEY_PREFIX)) return "mainnet";
    if (prefix.equals(window.TESTNET_M49_XPRIVATE_KEY_PREFIX)) return "testnet";
    if (prefix.equals(window.MAINNET_M84_XPRIVATE_KEY_PREFIX)) return "mainnet";
    if (prefix.equals(window.TESTNET_M84_XPRIVATE_KEY_PREFIX)) return "testnet";
    if (prefix.equals(window.MAINNET_XPUBLIC_KEY_PREFIX)) return "mainnet";
    if (prefix.equals(window.TESTNET_XPUBLIC_KEY_PREFIX)) return "testnet";
    if (prefix.equals(window.MAINNET_M49_XPUBLIC_KEY_PREFIX)) return "mainnet";
    if (prefix.equals(window.TESTNET_M49_XPUBLIC_KEY_PREFIX)) return "testnet";
    if (prefix.equals(window.MAINNET_M84_XPUBLIC_KEY_PREFIX)) return "mainnet";
    if (prefix.equals(window.TESTNET_M84_XPUBLIC_KEY_PREFIX)) return "testnet";
    throw new Error("invalid extended key");
  };

  window.xKeyType = (xKey) => {
    if (window.isString(xKey)) xKey = window.decodeBase58(xKey, { checkSum: true, hex: false });
    if (xKey.length !== 78) return false;
    let prefix = xKey.slice(0, 4);
    if (prefix.equals(window.MAINNET_XPRIVATE_KEY_PREFIX)) return "private";
    if (prefix.equals(window.TESTNET_XPRIVATE_KEY_PREFIX)) return "private";
    if (prefix.equals(window.MAINNET_M49_XPRIVATE_KEY_PREFIX)) return "private";
    if (prefix.equals(window.TESTNET_M49_XPRIVATE_KEY_PREFIX)) return "private";
    if (prefix.equals(window.MAINNET_M84_XPRIVATE_KEY_PREFIX)) return "private";
    if (prefix.equals(window.TESTNET_M84_XPRIVATE_KEY_PREFIX)) return "private";
    if (prefix.equals(window.MAINNET_XPUBLIC_KEY_PREFIX)) return "public";
    if (prefix.equals(window.TESTNET_XPUBLIC_KEY_PREFIX)) return "public";
    if (prefix.equals(window.MAINNET_M49_XPUBLIC_KEY_PREFIX)) return "public";
    if (prefix.equals(window.TESTNET_M49_XPUBLIC_KEY_PREFIX)) return "public";
    if (prefix.equals(window.MAINNET_M84_XPUBLIC_KEY_PREFIX)) return "public";
    if (prefix.equals(window.TESTNET_M84_XPUBLIC_KEY_PREFIX)) return "public";
    throw new Error("invalid extended key");
  };

  window.xKeyDerivationType = (xKey) => {
    if (window.isString(xKey)) xKey = window.decodeBase58(xKey, { checkSum: true, hex: false });
    if (xKey.length !== 78) return false;
    let prefix = xKey.slice(0, 4);
    if (prefix.equals(window.MAINNET_XPRIVATE_KEY_PREFIX)) return "BIP44";
    if (prefix.equals(window.TESTNET_XPRIVATE_KEY_PREFIX)) return "BIP44";
    if (prefix.equals(window.MAINNET_M49_XPRIVATE_KEY_PREFIX)) return "BIP49";
    if (prefix.equals(window.TESTNET_M49_XPRIVATE_KEY_PREFIX)) return "BIP49";
    if (prefix.equals(window.MAINNET_M84_XPRIVATE_KEY_PREFIX)) return "BIP84";
    if (prefix.equals(window.TESTNET_M84_XPRIVATE_KEY_PREFIX)) return "BIP84";
    if (prefix.equals(window.MAINNET_XPUBLIC_KEY_PREFIX)) return "BIP44";
    if (prefix.equals(window.TESTNET_XPUBLIC_KEY_PREFIX)) return "BIP44";
    if (prefix.equals(window.MAINNET_M49_XPUBLIC_KEY_PREFIX)) return "BIP49";
    if (prefix.equals(window.TESTNET_M49_XPUBLIC_KEY_PREFIX)) return "BIP49";
    if (prefix.equals(window.MAINNET_M84_XPUBLIC_KEY_PREFIX)) return "BIP84";
    if (prefix.equals(window.TESTNET_M84_XPUBLIC_KEY_PREFIX)) return "BIP84";
    return "custom";
  };

  window.pathXKeyTo_BIP32_XKey = (xKey, A = {}) => {
    ARGS(A, { base58: true });
    if (window.isString(xKey)) xKey = window.decodeBase58(xKey, { checkSum: true, hex: false });
    if (xKey.length !== 78) throw new Error("invalid extended key");
    let prefix = xKey.slice(0, 4);
    let newPrefix;
    if (prefix.equals(window.MAINNET_XPUBLIC_KEY_PREFIX)) newPrefix = prefix;
    else if (prefix.equals(window.TESTNET_XPUBLIC_KEY_PREFIX)) newPrefix = prefix;
    else if (prefix.equals(window.MAINNET_XPRIVATE_KEY_PREFIX)) newPrefix = prefix;
    else if (prefix.equals(window.TESTNET_XPRIVATE_KEY_PREFIX)) newPrefix = prefix;
    else if (prefix.equals(window.MAINNET_M49_XPUBLIC_KEY_PREFIX)) newPrefix = window.MAINNET_XPUBLIC_KEY_PREFIX;
    else if (prefix.equals(window.MAINNET_M84_XPUBLIC_KEY_PREFIX)) newPrefix = window.MAINNET_XPUBLIC_KEY_PREFIX;
    else if (prefix.equals(window.TESTNET_M49_XPUBLIC_KEY_PREFIX)) newPrefix = window.TESTNET_XPUBLIC_KEY_PREFIX;
    else if (prefix.equals(window.TESTNET_M84_XPUBLIC_KEY_PREFIX)) newPrefix = window.TESTNET_XPUBLIC_KEY_PREFIX;
    else if (prefix.equals(window.MAINNET_M49_XPRIVATE_KEY_PREFIX)) newPrefix = window.MAINNET_XPRIVATE_KEY_PREFIX;
    else if (prefix.equals(window.TESTNET_M49_XPRIVATE_KEY_PREFIX)) newPrefix = window.TESTNET_XPRIVATE_KEY_PREFIX;
    else if (prefix.equals(window.TESTNET_M84_XPRIVATE_KEY_PREFIX)) newPrefix = window.TESTNET_XPRIVATE_KEY_PREFIX;
    else if (prefix.equals(window.MAINNET_M84_XPRIVATE_KEY_PREFIX)) newPrefix = window.MAINNET_XPRIVATE_KEY_PREFIX;
    else throw new Error("invalid extended key");
    if (A.base58) return window.encodeBase58(BC([newPrefix, xKey.slice(4)]), { checkSum: true });
    return BC([newPrefix, xKey.slice(4)]);
  };

  window.BIP32_XKeyToPathXKey = (xKey, pathType, A = {}) => {
    ARGS(A, { base58: true });
    if (!["BIP44", "BIP49", "BIP84"].includes(pathType))
      throw new Error("unsupported path type " + pathType);
    if (window.isString(xKey)) xKey = window.decodeBase58(xKey, { checkSum: true, hex: false });
    if (xKey.length !== 78) throw new Error("invalid extended key");
    let prefix = xKey.slice(0, 4);
    let newPrefix;
    if (prefix.equals(window.TESTNET_XPRIVATE_KEY_PREFIX))
      switch (pathType) {
        case "BIP44": newPrefix = window.TESTNET_M44_XPRIVATE_KEY_PREFIX;
          break;
        case "BIP49": newPrefix = window.TESTNET_M49_XPRIVATE_KEY_PREFIX;
          break;
        case "BIP84": newPrefix = window.TESTNET_M84_XPRIVATE_KEY_PREFIX;
      }
    else if (prefix.equals(window.MAINNET_XPRIVATE_KEY_PREFIX))
      switch (pathType) {
        case "BIP44": newPrefix = window.MAINNET_M44_XPRIVATE_KEY_PREFIX;
          break;
        case "BIP49": newPrefix = window.MAINNET_M49_XPRIVATE_KEY_PREFIX;
          break;
        case "BIP84": newPrefix = window.MAINNET_M84_XPRIVATE_KEY_PREFIX;
      }
    else if (prefix.equals(window.TESTNET_XPUBLIC_KEY_PREFIX))
      switch (pathType) {
        case "BIP44": newPrefix = window.TESTNET_M44_XPUBLIC_KEY_PREFIX;
          break;
        case "BIP49": newPrefix = window.TESTNET_M49_XPUBLIC_KEY_PREFIX;
          break;
        case "BIP84": newPrefix = window.TESTNET_M84_XPUBLIC_KEY_PREFIX;
      }
    else if (prefix.equals(window.MAINNET_XPUBLIC_KEY_PREFIX))
      switch (pathType) {
        case "BIP44": newPrefix = window.MAINNET_M44_XPUBLIC_KEY_PREFIX;
          break;
        case "BIP49": newPrefix = window.MAINNET_M49_XPUBLIC_KEY_PREFIX;
          break;
        case "BIP84": newPrefix = window.MAINNET_M84_XPUBLIC_KEY_PREFIX;
      }
    else throw new Error("invalid extended key");
    if (A.base58) return window.encodeBase58(BC([newPrefix, xKey.slice(4)]), { checkSum: true });
    return BC([newPrefix, xKey.slice(4)]);
  };


}

function mnemonic(window) {
  let BN = window.BN;
  let C = window.__nodeCrypto;
  let W = window.getWindow();
  let ARGS = window.defArgs;


  window.getRandomValues = (buf) => {
    if (W.crypto && W.crypto.getRandomValues) return W.crypto.getRandomValues(buf);
    if (typeof W.msCrypto === 'object' && typeof W.msCrypto.getRandomValues === 'function')
      return W.msCrypto.getRandomValues(buf);

    if (C !== false) {
      if (!(buf instanceof Uint8Array)) throw new TypeError('expected Uint8Array');
      if (buf.length > 65536) {
        let e = new Error();
        e.code = 22;
        e.message = 'Failed to execute \'getRandomValues\' on \'Crypto\': The ' +
          'ArrayBufferView\'s byte length (' + buf.length + ') exceeds the ' +
          'number of bytes of entropy available via this API (65536).';
        e.name = 'QuotaExceededError';
        throw e;
      }
      let bytes = C.randomBytes(buf.length);
      buf.set(bytes);
      return buf;
    } else throw new Error('No secure random number generator available.');
  };

  window.lngamma = (z) => {
    if (z < 0) return null;
    let x = window.GAMMA_TABLE_LN[0];
    for (let i = window.GAMMA_TABLE_LN.length - 1; i > 0; --i) x += window.GAMMA_TABLE_LN[i] / (z + i);
    let t = z + window.GAMMA_NUM_LN + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x) - Math.log(z);
  };

  window.igam = (a, x) => {
    if (x <= 0 || a <= 0) return 0.0;
    if (x > 1.0 && x > a) return 1.0 - window.igamc(a, x);
    let ans, ax, c, r;
    /* Compute xa exp(-x) / gamma(a) */
    ax = a * Math.log(x) - x - window.lngamma(a);
    if (ax < -window.MAXLOG) return (0.0);
    ax = Math.exp(ax);
    /* power series */
    r = a;
    c = 1.0;
    ans = 1.0;

    do {
      r += 1.0;
      c *= x / r;
      ans += c;
    } while (c / ans > window.MACHEP);

    return (ans * ax / a);
  };

  window.igamc = (a, x) => {
    if (x <= 0 || a <= 0) return 1.0;
    if (x < 1.0 || x < a) return 1.0 - igam(a, x);
    let big = 4.503599627370496e15;
    let biginv = 2.22044604925031308085e-16;
    let ans, ax, c, yc, r, t, y, z;
    let pk, pkm1, pkm2, qk, qkm1, qkm2;
    ax = a * Math.log(x) - x - window.lngamma(a);
    if (ax < -window.MAXLOG) return 0.0;
    ax = Math.exp(ax);
    y = 1.0 - a;
    z = x + y + 1.0;
    c = 0.0;
    pkm2 = 1.0;
    qkm2 = x;
    pkm1 = x + 1.0;
    qkm1 = z * x;
    ans = pkm1 / qkm1;

    do {
      c += 1.0;
      y += 1.0;
      z += 2.0;
      yc = y * c;
      pk = pkm1 * z - pkm2 * yc;
      qk = qkm1 * z - qkm2 * yc;
      if (qk !== 0) {
        r = pk / qk;
        t = Math.abs((ans - r) / r);
        ans = r;
      } else t = 1.0;

      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;
      if (Math.abs(pk) > big) {
        pkm2 *= biginv;
        pkm1 *= biginv;
        qkm2 *= biginv;
        qkm1 *= biginv;
      }
    } while (t > window.MACHEP);

    return ans * ax;
  };

  window.erfc = (x) => {
    let z = Math.abs(x);
    let t = 1 / (1 + z / 2);
    let r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 +
      t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
        t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
          t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  };

  window.randomnessTest = (b) => {
    // NIST SP 800-22 randomness tests
    // https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf

    let p = new BN(b);
    let s = p.toString(2).padStart(256, '0');
    // Frequency (Monobit) Test
    let n = s.length;
    let s_0 = (s.match(/0/g) || []).length;
    let s_1 = (s.match(/1/g) || []).length;
    let s_obs = Math.abs(s_1 - s_0) / Math.sqrt(2 * n);
    if (!(window.erfc(s_obs) > 0.01)) throw new Error('Frequency (Monobit) Test failed.');

    // Runs Test
    let pi = s_1 / n;
    if (!(Math.abs(pi - 0.5) < 2 / Math.sqrt(n))) throw new Error('Runs Test failed.');
    let v = 1;
    for (let i = 0; i < n - 1; i++) v += (s[i] === s[i + 1]) ? 0 : 1;
    let a = v - 2 * n * pi * (1 - pi);
    let q = 2 * Math.sqrt(2 * n) * pi * (1 - pi);
    if (!(window.erfc(Math.abs(a) / q) > 0.01)) throw new Error('Runs Test failed.');

    // Test for the Longest Run of Ones in a Block
    s = [s.substring(0, 128).match(/.{1,8}/g), s.substring(128, 256).match(/.{1,8}/g)];
    for (let w = 0; w < 2; w++) {
      let sl = s[w];
      v = [0, 0, 0, 0];
      for (let i = 0; i < sl.length; i++) {
        let q = sl[i].split('0');
        let l = q.reduce(function (a, b) {
          return a.length > b.length ? a : b;
        }).length;
        switch (l) {
          case 0:
            v[0] += 1;
            break;
          case 1:
            v[0] += 1;
            break;
          case 2:
            v[1] += 1;
            break;
          case 3:
            v[2] += 1;
            break;
          default:
            v[3] += 1;
        }
      }

      let k = 3;
      let r = 16;
      pi = [0.2148, 0.3672, 0.2305, 0.1875];
      let x_sqrt = Math.pow(v[0] - r * pi[0], 2) / (r * pi[0]);
      x_sqrt += Math.pow(v[1] - r * pi[1], 2) / (r * pi[1]);
      x_sqrt += Math.pow(v[2] - r * pi[2], 2) / (r * pi[2]);
      x_sqrt += Math.pow(v[3] - r * pi[3], 2) / (r * pi[3]);
      if (!(window.igamc(k / 2, x_sqrt / 2) > 0.01))
        throw new Error('Test for the Longest Run of Ones in a Block failed.');
    }
  };

  /**
  * Generate 128-256 bits entropy bytes string
  *
  * :param strength: (optional) entropy bits strength, by default is 256 bit.
  * :param hex: (optional) return HEX encoded string result flag, by default is ``true``.
  * :param sec256k1Order: (optional) if ``true`` ECDSA_SEC256K1_ORDER, by default is ``true``.
  * :return: HEX encoded or bytes entropy string.
  */
  window.generateEntropy = (A = {}) => {
    ARGS(A, { strength: 256, hex: true, sec256k1Order: true });
    if (!([128, 160, 192, 224, 256].includes(A.strength)))
      throw new TypeError('strength should be one of the following [128, 160, 192, 224, 256]');

    let b = window.Buffer.alloc(32);
    let attempt = 0, p, f;
    do {
      f = true;
      if (attempt++ > 100) throw new Error('Generate randomness failed');
      window.getRandomValues(b);
      if (A.sec256k1Order) {
        p = new BN(b);
        if ((p.gte(window.ECDSA_SEC256K1_ORDER))) continue;
      }
      try { window.randomnessTest(b); } catch (e) { f = false; }
    }
    while (!f);
    b = b.slice(0, A.strength / 8);
    return A.hex ? b.hex() : b;
  };


  /**
  * Convert entropy to mnemonic words string.
  *
  * :parameters:
  *   :e: random entropy HEX encoded or bytes string.
  * :param wordList: (optional) word list, by default is BIP39_WORDLIST
  * :return: mnemonic words string.
  */
  window.entropyToMnemonic = (e, A = {}) => {
    ARGS(A, { wordList: window.BIP39_WORDLIST, data: false });
    e = window.getBuffer(e);
    let i = new BN(e, 16);
    if (!([16, 20, 24, 28, 32].includes(e.length)))
      throw new TypeError('entropy length should be one of the following: [16, 20, 24, 28, 32]');
    if (!(A.wordList instanceof Array) || (A.wordList.length !== 2048))
      throw new TypeError('invalid wordlist');

    let b = Math.ceil(e.length * 8 / 32);

    if (A.data !== false) {
      if (A.data > (2 ** b - 1)) throw new TypeError('embedded data bits too long');
      i = i.shln(b).or(new BN(A.data));

    } else i = i.shln(b).or(new BN(window.sha256(e)[0] >> (8 - b)));
    let r = [];
    for (let d = (e.length * 8 + 8) / 11 | 0; d > 0; d--)
      r.push(A.wordList[i.shrn((d - 1) * 11).and(new BN(2047)).toNumber()]);
    return r.join(' ');
  };

  /**
  * Converting mnemonic words to entropy.
  *
  * :parameters:
  *   m: mnemonic words string (space separated)
  * :param wordList: (optional) word list, by default is BIP39_WORDLIST
  * :param checkSum: (optional) boolean, by default is``false``.
  * :param hex: (optional) return HEX encoded string result flag, by default is ``false``.
  * :return: HEX encoded or bytes string.
  */
  window.mnemonicToEntropy = (m, A = {}) => {
    ARGS(A, { wordList: window.BIP39_WORDLIST, checkSum: false, hex: true });
    m = m.trim().split(/\s+/);
    if (!(window.isMnemonicValid(m, A))) throw new TypeError('invalid mnemonic words');
    let e = new BN(0);
    for (let w of m) e = e.shln(11).or(new BN(A.wordList.indexOf(w)));
    let bitSize = m.length * 11;
    let checkSumBitLen = bitSize % 32;
    e = e.shrn(checkSumBitLen);
    e = e.toArrayLike(window.Buffer, 'be', Math.ceil((bitSize - checkSumBitLen) / 8));
    return (A.hex) ? e.hex() : e;
  };


  /**
  * Converting mnemonic words string to seed for uses in key derivation (BIP-0032).
  *
  * :parameters:
  *   :m: mnemonic words string (space separated)
  * :param passphrase: (optional) passphrase to get ability use 2FA approach for creating seed, by default is empty string.
  * :param checkSum: (optional) boolean, by default is ``false``.
  * :param hex: (optional) return HEX encoded string result flag, by default is ``true``.
  * :return: HEX encoded or bytes string.
  */
  window.mnemonicToSeed = (m, A = {}) => {
    ARGS(A, { passphrase: "", checkSum: false, hex: true });
    if (!window.isString(m)) throw new Error("mnemonic should be string");
    if (!window.isString(A.passphrase)) throw new Error("passphrase should be string");
    let s = window.pbdkdf2HmacSha512(m, "mnemonic" + A.passphrase, 2048);
    return (A.hex) ? s.hex() : s;
  };

  window.isMnemonicValid = (m, A = {}) => {
    ARGS(A, { wordList: window.BIP39_WORDLIST });
    if (window.isString(m)) m = m.trim().split(/\s+/);
    for (let w of m) if (!(A.wordList.includes(w))) return false;
    return true
  };

  window.isMnemonicCheckSumValid = (m, A = {}) => {
    ARGS(A, { wordList: window.BIP39_WORDLIST });
    let e;
    try {
      e = window.mnemonicToEntropy(m, { wordList: A.wordList, hex: false });
    } catch (e) {
      return false;
    }
    m = m.trim().split(/\s+/);
    let bitSize = m.length * 11;
    let checkSumBitLen = bitSize % 32;
    let c = window.sha256(e)[0] >> (8 - checkSumBitLen);
    let c2 = window.intToBytes(A.wordList.indexOf(m.pop()), 1) & (2 ** checkSumBitLen - 1);
    return c === c2;
  };

  window.getMnemonicCheckSumData = (m, A = {}) => {
    ARGS(A, { wordList: window.BIP39_WORDLIST });
    m = m.trim().split(/\s+/);
    let bitSize = m.length * 11;
    let checkSumBitLen = bitSize % 32;
    return window.intToBytes(A.wordList.indexOf(m.pop()), 1) & (2 ** checkSumBitLen - 1);
  };

  window.__combinations = (a, n) => {
    let results = [], i;
    let total = Math.pow(2, a.length);
    for (let m = n; m < total; m++) {
      let r = [];
      i = a.length - 1;

      do {
        if ((m & (1 << i)) !== 0) r.push(a[i]);
      } while (i--);

      if (r.length >= n) {
        results.push(r);
      }
    }
    return results;
  };

  window.splitMnemonic = (threshold, total, m, A = {}) => {
    ARGS(A, {
      wordList: window.BIP39_WORDLIST,
      checkSumVerify: false,
      sharesVerify: false,
      embeddedIndex: false,
      hex: true
    });
    let e = window.mnemonicToEntropy(m, {
      wordList: A.wordList,
      checkSum: A.checkSumVerify, hex: false
    });
    let bits;
    if (A.embeddedIndex)
      bits = Math.ceil(Math.log2(total)) + 1;
    else
      bits = 8;

    let shares = window.__split_secret(threshold, total, e, bits);


    if (A.sharesVerify) {
      // shares validation
      let a = [];
      for (let i in shares) {
        i = parseInt(i);
        a.push([i, shares[i]]);
      }
      let combinations = window.__combinations(a, threshold);
      for (let c of combinations) {
        let d = {};
        for (let q of c) d[q[0]] = q[1];
        let s = window.__restore_secret(d);
        if (!s.equals(e)) {
          throw new Error("split secret failed");
        }
      }
    }


    let result;

    if (A.embeddedIndex === false) {
      result = {};
      for (let i in shares) result[i] = window.entropyToMnemonic(shares[i], A);
    } else {
      result = [];
      for (let i in shares) {
        A.data = i;
        result.push(window.entropyToMnemonic(shares[i], A));
      }
    }

    return result;
  };

  window.combineMnemonic = (shares, A = {}) => {
    let embeddedIndex = shares.constructor === Array;
    let s = {};
    if (embeddedIndex) {
      for (let share of shares) {
        let e = window.mnemonicToEntropy(share, {
          wordList: A.wordList,
          checkSum: false,
          hex: false
        });
        let i = window.getMnemonicCheckSumData(share);
        if (s[i] !== undefined) throw new Error("Non unique or invalid shares");
        s[i] = e;
      }
    } else {
      for (let i in shares) s[i] = window.mnemonicToEntropy(shares[i],
        {
          wordList: A.wordList,
          checkSum: A.checkSum, hex: false
        });
    }

    return window.entropyToMnemonic(window.__restore_secret(s), A);
  };

}

function encoders(window) {
  let CM = window.__bitcoin_core_crypto.module;
  let BA = Buffer.alloc;
  let BC = Buffer.concat;
  let getBuffer = window.getBuffer;
  let ARGS = window.defArgs;
  let malloc = CM._malloc;
  let free = CM._free;
  let getValue = CM.getValue;

  /**
  * Encode to a base58-encoded string
  *
  * :parameters:
  *   :m: string to encode
  * :param encoding: (optional), by default 'hex|utf8'.
  * :param checksum: (optional) boolean, by default is ``false``.
  * :return: base58 string format.
  */
  window.encodeBase58 = (m, A = {}) => {
    ARGS(A, { encoding: 'hex|utf8', checkSum: false });
    m = getBuffer(m, A.encoding);
    if (A.checkSum) m = BC([m, window.doubleSha256(m).slice(0, 4)]);
    if (m.length > 1073741823) throw new Error('encodeBase58 message is too long');

    let bP = malloc(m.length);
    let eS = m.length * 138 / 100 + 1;
    let oP = malloc(m.length * 138 / 100 + 1);
    CM.HEAPU8.set(m, bP);
    CM._EncodeBase58(bP, bP + m.length, oP);
    let out = new BA(eS);
    let q;
    for (q = 0; q <= eS; q++) {
      out[q] = getValue(oP + q, 'i8');
      if (out[q] === 0) break
    }
    free(bP);
    free(oP);
    return out.slice(0, q).toString();
  };

  /**
  * Decode a base58-encoding string
  *
  * :parameters:
  *   :m: string to decode
  * :param hex: (optional) return result as HEX encoded string, by default is ``true``.
  * :param checksum: (optional) boolean, by default is ``false``.
  * :return: HEX or bytes string format.
  */
  window.decodeBase58 = (m, A = {}) => {
    ARGS(A, { hex: true, checkSum: false });
    if (!window.isString(m)) throw new Error('decodeBase58 string required');
    if (m.length > 2147483647) throw new Error('decodeBase58 string is too long');
    let mB = new BA(m.length + 1);
    mB.write(m);
    mB.writeInt8(0, m.length);
    let bP = malloc(mB.length);
    let oLP = malloc(4);
    let oP = malloc(Math.ceil(m.length * 733 / 1000) + 2);
    CM.HEAPU8.set(mB, bP);
    let r = CM._DecodeBase58(bP, oP, oLP);
    free(bP);
    if (r) {
      let oL = CM.getValue(oLP, 'i32');
      free(oLP);
      let out = new BA(oL);
      for (let q = 0; q <= oL; q++) out[q] = getValue(oP + q, 'i8');
      free(oP);
      if (A.checkSum) out = out.slice(0, -4);
      return (A.hex) ? out.hex() : out;
    }
    free(oP);
    free(oLP);
    return "";
  };

  window.rebaseBits = (data, fromBits, toBits, pad) => {
    if (pad === undefined) pad = true;
    let acc = 0, bits = 0, ret = [];
    let maxv = (1 << toBits) - 1;
    let max_acc = (1 << (fromBits + toBits - 1)) - 1;
    for (let i = 0; i < data.length; i++) {
      let value = data[i];
      if ((value < 0) || (value >> fromBits)) throw ("invalid bytes");
      acc = ((acc << fromBits) | value) & max_acc;
      bits += fromBits;
      while (bits >= toBits) {
        bits -= toBits;
        ret.push((acc >> bits) & maxv);
      }
    }
    if (pad === true) {
      if (bits)
        ret.push((acc << (toBits - bits)) & maxv);
    } else if ((bits >= fromBits) || ((acc << (toBits - bits)) & maxv))
      throw ("invalid padding");
    return ret
  };

  window.rebase_5_to_8 = function (data, pad) {
    if (pad === undefined) pad = true;
    return window.rebaseBits(data, 5, 8, pad);
  };

  window.rebase_8_to_5 = (data, pad) => {
    if (pad === undefined) pad = true;
    return window.rebaseBits(data, 8, 5, pad);
  };

  window.rebase_32_to_5 = (data) => {
    if (typeof (data) !== "string") data = window.bytesToString(data);
    let b = [];
    try {
      for (let i = 0; i < data.length; i++) b.push(window.INT_BASE32_MAP[data[i]]);
    } catch (err) {
      throw ("Non base32 characters");
    }
    return b;
  };

  window.rebase_5_to_32 = (data, bytes) => {
    if (bytes === undefined) bytes = true;
    let r = [];
    for (let i = 0; i < data.length; i++) r.push(window.BASE32_INT_MAP[data[i]]);
    return (bytes === true) ? r : window.bytesToString(r);
  };

  window.bech32Polymod = (values) => {
    let generator = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];
    let chk = 1;
    for (let i = 0; i < values.length; i++) {
      let top = chk >> 25;
      chk = (chk & 0x1ffffff) << 5 ^ values[i];
      for (let k = 0; k < 5; k++) {
        if ((top >> k) & 1) chk ^= generator[k];
        else chk ^= 0;
      }
    }
    return chk ^ 1;
  };
}

function hash(window) {
  let CM = window.__bitcoin_core_crypto.module;
  let malloc = CM._malloc;
  let free = CM._free;
  let BA = window.Buffer.alloc;
  let ARGS = window.defArgs;
  let getBuffer = window.getBuffer;
  let BN = window.BN;
  let getValue = CM.getValue;

  window.sha256 = (m, A = {}) => {
    ARGS(A, { encoding: 'hex|utf8', hex: false });
    m = getBuffer(m, A.encoding);
    let bP = malloc(m.length);
    let oP = malloc(32);
    CM.HEAPU8.set(m, bP);
    CM._single_sha256(bP, m.length, oP);
    let out = new BA(32);
    for (let i = 0; i < 32; i++) out[i] = getValue(oP + i, 'i8');
    free(bP);
    free(oP);
    return (A.hex) ? out.hex() : out;
  };

  window.SHA256 = class {
    constructor() {
      this.ptr = new CM.CSHA256();
      this.result = new BA(32);
      return this;
    };
    update(m, A = {}) {
      ARGS(A, { encoding: 'hex|utf8' });
      m = getBuffer(m, A.encoding);
      let bP = malloc(m.length);
      CM.HEAPU8.set(m, bP);
      this.ptr.Write(bP, m.length);
      free(bP);
      return this;
    };
    digest() {
      let oP = malloc(32);
      this.ptr.Finalize(oP);
      for (let i = 0; i < 32; i++) this.result[i] = getValue(oP + i, 'i8');
      free(oP);
      return this.result;
    };
    hexdigest() {
      let oP = malloc(32);
      this.ptr.Finalize(oP);
      for (let i = 0; i < 32; i++) this.result[i] = getValue(oP + i, 'i8');
      free(oP);
      return this.result.hex();
    }
  };

  window.doubleSha256 = (m, A = {}) => {
    ARGS(A, { encoding: 'hex|utf8', hex: false });
    m = getBuffer(m, A.encoding);
    let bP = malloc(m.length);
    let oP = malloc(32);
    CM.HEAPU8.set(m, bP);
    CM._double_sha256(bP, m.length, oP);
    let out = new BA(32);
    for (let i = 0; i < 32; i++) out[i] = getValue(oP + i, 'i8');
    free(bP);
    free(oP);
    return (A.hex) ? out.hex() : out;
  };

  window.siphash = (m, A = {}) => {
    ARGS(A, { encoding: 'hex|utf8', v0: window.BNZerro, v1: window.BNZerro });
    if (!(A.v1 instanceof BN) || !(A.v0 instanceof BN)) throw new Error('siphash init vectors v0, v1 must be BN instance');
    m = getBuffer(m, A.encoding);
    let v0b = A.v0.toArrayLike(Uint8Array, 'le', 8);
    let v1b = A.v1.toArrayLike(Uint8Array, 'le', 8);
    let bP = malloc(m.length);
    let v0Pointer = malloc(8);
    let v1Pointer = malloc(8);
    let oP = malloc(8);
    CM.HEAPU8.set(m, bP);
    CM.HEAPU8.set(v0b, v0Pointer);
    CM.HEAPU8.set(v1b, v1Pointer);
    CM._siphash(v0Pointer, v1Pointer, bP, m.length, oP);
    let out = new BA(9);
    for (let i = 0; i < 8; i++) out[8 - i] = getValue(oP + i, 'i8');
    free(bP);
    free(oP);
    return new BN(out);
  };

  window.ripemd160 = (m, A = {}) => {
    ARGS(A, { encoding: 'hex|utf8', hex: false });
    m = getBuffer(m, A.encoding);
    let bP = malloc(m.length);
    let oP = malloc(32);
    CM.HEAPU8.set(m, bP);
    CM.__ripemd160(bP, m.length, oP);
    let out = new BA(20);
    for (let i = 0; i < 20; i++) out[i] = getValue(oP + i, 'i8');
    free(bP);
    free(oP);
    return (A.hex) ? out.hex() : out;
  };

  window.md5 = (m, A = {}) => {
    ARGS(A, { encoding: 'hex|utf8', hex: false });
    m = getBuffer(m, A.encoding);
    let bP = malloc(m.length);
    let oP = malloc(16);
    CM.HEAPU8.set(m, bP);
    CM._md5sum(bP, m.length, oP);
    let out = new BA(16);
    for (let i = 0; i < 16; i++) out[i] = getValue(oP + i, 'i8');
    free(bP);
    free(oP);
    return (A.hex) ? out.hex() : out;
  };

  window.sha3 = (m, A = {}) => {
    ARGS(A, { encoding: 'hex|utf8', hex: false });
    m = getBuffer(m, A.encoding);
    let bP = malloc(m.length);
    let oP = malloc(32);
    CM.HEAPU8.set(m, bP);
    CM._sha3(bP, m.length, oP);
    let out = new BA(32);
    for (let i = 0; i < 32; i++) out[i] = getValue(oP + i, 'i8');
    free(bP);
    free(oP);
    return (A.hex) ? out.hex() : out;
  };


  window.hash160 = (m, A = {}) => {
    ARGS(A, { encoding: 'hex|utf8', hex: false });
    return window.ripemd160(window.sha256(m, { hex: false, encoding: A.encoding }), { hex: A.hex });
  };

  window.hmacSha512 = (k, d, A = {}) => {
    ARGS(A, { encoding: 'hex|utf8', hex: false });
    k = getBuffer(k, A.encoding);
    d = getBuffer(d, A.encoding);
    let kP = malloc(k.length);
    let dP = malloc(d.length);
    let oP = malloc(64);
    CM.HEAPU8.set(k, kP);
    CM.HEAPU8.set(d, dP);
    CM._hmac_sha512_oneline(kP, k.length, dP, d.length, oP);
    let out = new BA(64);
    for (let i = 0; i < 64; i++) out[i] = getValue(oP + i, 'i8');
    free(kP);
    free(dP);
    free(oP);
    return (A.hex) ? out.hex() : out;
  };

  window.pbdkdf2HmacSha512 = (password, salt, i, A = {}) => {
    ARGS(A, { encoding: 'utf8', hex: false });
    let p = getBuffer(password, A.encoding);
    let s = getBuffer(salt, A.encoding);
    let pP = malloc(p.length);
    let sP = malloc(s.length);
    let oP = malloc(64);
    CM.HEAPU8.set(p, pP);
    CM.HEAPU8.set(s, sP);
    CM._pbkdf2_hmac_sha512(pP, p.length, sP, s.length, i, oP, 64);
    let out = new BA(64);
    for (let i = 0; i < 64; i++) out[i] = getValue(oP + i, 'i8');
    free(pP);
    free(sP);
    free(oP);
    return (A.hex) ? out.hex() : out;
  };


}

function key(window) {
  let Buffer = window.Buffer;
  let BF = Buffer.from;
  let BC = Buffer.concat;
  let BA = Buffer.alloc;
  let isBuffer = window.isBuffer;
  let getBuffer = window.getBuffer;
  let ARGS = window.defArgs;
  let crypto = window.__bitcoin_core_crypto.module;
  let malloc = crypto._malloc;
  let free = crypto._free;
  let getValue = crypto.getValue;


  /**
  * Create private key
  *
  * :param compressed: (optional) flag of private key compressed format, by default is ``true``
  * :param testnet: (optional) flag for testnet network, by default is ``false``.
  * :param wif:  (optional) If set to ``true`` return key in WIF format, by default is ``true``.
  * :param hex:  (optional) If set to ``true`` return key in HEX format, by default is ``false``.
  * :return: Private key in wif format (default), hex encoded byte string in case of hex flag or raw bytes string in case wif and hex flags set to ``false``.
  */
  window.createPrivateKey = (A = {}) => {
    ARGS(A, { compressed: true, testnet: false, wif: true, hex: false });
    if (A.wif) return window.privateKeyToWif(window.generateEntropy({ hex: false }), A);
    if (A.hex) return window.generateEntropy({ hex: true });
    return window.generateEntropy({ hex: false });
  };

  /**
  * Encode private key in HEX or RAW bytes format to WIF format.
  *
  * :parameters:
  *   :h: private key 32 byte string or HEX encoded string.
  * :param compressed: (optional) flag of public key compressed format, by default is ``true``.
  * :param testnet: (optional) flag for testnet network, by default is ``false``.
  * :return: Private key in WIF format.
  */
  window.privateKeyToWif = (h, A = {}) => {
    ARGS(A, { compressed: true, testnet: false });
    h = getBuffer(h);
    if (h.length !== 32) throw new Error('invalid byte string');
    let prefix;
    if (A.testnet) prefix = BF(window.TESTNET_PRIVATE_KEY_BYTE_PREFIX);
    else prefix = BF(window.MAINNET_PRIVATE_KEY_BYTE_PREFIX);

    if (A.compressed) h = BC([prefix, h, Buffer.from([1])]);
    else h = BC([prefix, h]);

    h = BC([h, window.doubleSha256(h).slice(0, 4)]);
    return window.encodeBase58(h);
  };

  /**
  * Decode WIF private key to bytes string or HEX encoded string
  *
  * :parameters:
  *   :h: private key in WIF format string.
  * :param hex:  (optional) if set to ``true`` return key in HEX format, by default is ``true``.
  * :return: Private key HEX encoded string or raw bytes string.
  */
  window.wifToPrivateKey = (h, A = {}) => {
    ARGS(A, { hex: true });
    h = window.decodeBase58(h, { hex: false });
    if (!window.doubleSha256(h.slice(0, h.length - 4), { hex: false }).slice(0, 4).equals(h.slice(h.length - 4, h.length)))
      throw new Error('invalid byte string');
    return (A.hex) ? h.slice(1, 33).hex() : h.slice(1, 33)
  };


  /**
  Check is private key in WIF format string is valid.
  *
  * :parameters:
  *   :wif: private key in WIF format string.
  * :return: boolean.
  */
  window.isWifValid = (wif) => {
    if (!window.isString(wif)) return false;
    if (!window.PRIVATE_KEY_PREFIX_LIST.includes(wif[0])) return false;
    try {
      let h = window.decodeBase58(wif, { hex: false });
      let checksum = h.slice(h.length - 4, h.length);
      let unc = [window.MAINNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX,
      window.TESTNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX];
      if (unc.includes(wif[0])) {
        if (h.length !== 37) return false;
      } else {
        if (h.length !== 38) return false;
      }
      let calcChecksum = window.doubleSha256(h.slice(0, h.length - 4), { hex: false }).slice(0, 4);
      return calcChecksum.equals(checksum);
    } catch (e) {
    }
    return false;
  };

  /**
  * Get public key from private key using ECDSA secp256k1
  *
  * :parameters:
  *   :privateKey: private key in WIF, HEX or bytes.
  * :param compressed: (optional) flag of public key compressed format, by default is ``true``. In case private_key in WIF format, this flag is set in accordance with the key format specified in WIF string.
  * :param hex:  (optional) if set to ``true`` return key in HEX format, by default is ``true``.
  * :return: 33/65 bytes public key in HEX or bytes string.
  */
  window.privateToPublicKey = (privateKey, A = {}) => {
    ARGS(A, { compressed: true, hex: true });
    if (!isBuffer(privateKey)) {
      if (window.isString(privateKey)) {
        if (window.isHex(privateKey)) privateKey = Buffer.from(privateKey, 'hex');
        else {
          let unc = [window.MAINNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX,
          window.TESTNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX];
          if (unc.includes(privateKey[0])) A.compressed = false;
          privateKey = window.wifToPrivateKey(privateKey, { hex: false });
        }
      } else {
        throw new Error('invalid private key string');
      }
    }
    if (privateKey.length !== 32) throw new Error('private key length invalid');
    let privateKeyPointer = malloc(32);
    let publicKeyPointer = malloc(64);
    crypto.HEAPU8.set(privateKey, privateKeyPointer);
    crypto._secp256k1_ec_pubkey_create(window.secp256k1PrecompContextSign, publicKeyPointer, privateKeyPointer);
    free(privateKeyPointer);
    let outq = new BA(64);
    for (let i = 0; i < 64; i++) outq[i] = getValue(publicKeyPointer + i, 'i8');
    let pubLen = (A.compressed) ? 33 : 65;
    let publicKeySerializedPointer = malloc(65);
    let pubLenPointer = malloc(1);
    crypto.HEAPU8.set([pubLen], pubLenPointer);
    let flag = (A.compressed) ? window.SECP256K1_EC_COMPRESSED : window.SECP256K1_EC_UNCOMPRESSED;
    let r = crypto._secp256k1_ec_pubkey_serialize(window.secp256k1PrecompContextVerify,
      publicKeySerializedPointer, pubLenPointer, publicKeyPointer, flag);
    let out;
    if (r) {
      out = new BA(pubLen);
      for (let i = 0; i < pubLen; i++) out[i] = getValue(publicKeySerializedPointer + i, 'i8');
    } else out = false;

    free(publicKeyPointer);
    free(pubLenPointer);
    free(publicKeySerializedPointer);
    if (out === false) throw new Error('privateToPublicKey failed');
    return (A.hex) ? out.hex() : out;
  };

  /**
  * Check public key is valid.
  *
  * :parameters:
  *   :key: public key in HEX or bytes string format.
  * :return: boolean.
  */
  window.isPublicKeyValid = (key) => {
    if (window.isString(key)) {
      if (!window.isHex(key)) return false;
      key = BF(key, 'hex');
    }
    if (key.length < 33) return false;
    if ((key[0] === 4) && (key.length !== 65)) return false;
    if ((key[0] === 2) || (key[0] === 3))
      if (key.length !== 33) return false;
    return !((key[0] < 2) || (key[0] > 4));
  };

  window.publicKeyAdd = (key, tweak, A = {}) => {
    ARGS(A, { compressed: true, hex: true });
    key = window.getBuffer(key);
    tweak = window.getBuffer(tweak);
    let keyP = malloc(65);
    let tweakP = malloc(tweak.length);
    crypto.HEAPU8.set(key, keyP);
    crypto.HEAPU8.set(tweak, tweakP);
    let rawKeyP = malloc(65);

    let r = crypto._secp256k1_ec_pubkey_parse(window.secp256k1PrecompContextVerify, rawKeyP, keyP, key.length);


    if (!r) throw new Error('publicKeyAdd failed');
    r = crypto._secp256k1_ec_pubkey_tweak_add(window.secp256k1PrecompContextVerify, rawKeyP, tweakP);
    free(tweakP);

    if (!r) throw new Error('publicKeyAdd failed');
    let flag = (A.compressed) ? window.SECP256K1_EC_COMPRESSED : window.SECP256K1_EC_UNCOMPRESSED;
    let pubLen = (A.compressed) ? 33 : 65;
    let publicKeySerializedPointer = malloc(65);
    let pubLenPointer = malloc(1);
    crypto.HEAPU8.set([pubLen], pubLenPointer);
    r = crypto._secp256k1_ec_pubkey_serialize(window.secp256k1PrecompContextVerify,
      publicKeySerializedPointer, pubLenPointer, rawKeyP, flag);
    free(rawKeyP);
    free(keyP);

    let out;
    if (r) {
      out = new BA(pubLen);
      for (let i = 0; i < pubLen; i++) out[i] = getValue(publicKeySerializedPointer + i, 'i8');
    } else out = false;

    free(pubLenPointer);
    free(publicKeySerializedPointer);
    if (out === false) throw new Error('publicKeyAdd failed');
    return (A.hex) ? out.hex() : out;
  };
}

function script(window) {
  let Buffer = window.Buffer;
  let ARGS = window.defArgs;
  let getBuffer = window.getBuffer;
  let BF = Buffer.from;
  let BC = Buffer.concat;
  let O = window.OPCODE;
  let RO = window.RAW_OPCODE;
  let CM = window.__bitcoin_core_crypto.module;
  let malloc = CM._malloc, free = CM._free, getValue = CM.getValue;

  window.hashToScript = (h, scriptType, A = {}) => {
    ARGS(A, { hex: false });
    if (window.isString(scriptType)) scriptType = window.SCRIPT_TYPES[scriptType];
    h = getBuffer(h);
    let s;
    switch (scriptType) {
      case 0:
        s = BC([BF([O.OP_DUP, O.OP_HASH160, 0x14]), h, BF([O.OP_EQUALVERIFY, O.OP_CHECKSIG])]);
        break;
      case 1:
        s = BC([BF([O.OP_HASH160, 0x14]), h, BF([O.OP_EQUAL])]);
        break;
      case 5:
      case 6:
        s = BC([BF([0, 0x14]), h]);
        break;
      default:
        throw new Error('unsupported script type');
    }
    return (A.hex) ? s.hex() : s;
  };

  window.publicKeyTo_P2SH_P2WPKH_Script = (h, A = {}) => {
    ARGS(A, { hex: false });
    h = getBuffer(h);
    if (h.length !== 33) throw new Error("public key len invalid");
    let s = BC([BF([0, 0x14]), window.hash160(h)]);
    return (A.hex) ? s.hex() : s;
  };

  window.publicKeyTo_PUBKEY_Script = (k, A = {}) => {
    ARGS(A, { hex: false });
    k = getBuffer(k);
    let s = BC([BF([k.length]), k, BF([O.OP_CHECKSIG])]);
    return (A.hex) ? s.hex() : s;
  };

  /**
  * Parse script and return script type, script address and required signatures count.
  *
  * :parameters:
  *   :s: script in bytes string or HEX encoded string format.
  * :param segwit:  (optional) If set to ``true`` recognize P2WPKH and P2WSH sripts, by default is ``true``.
  * :return: object:
  *
  *        - nType - numeric script type
  *        - type  - script type
  *        - addressHash - address hash in case address recognized
  *        - script - script if no address recognized
  *        - reqSigs - required signatures count
  */
  window.parseScript = (s, A = {}) => {
    ARGS(A, { segwit: true });
    s = getBuffer(s);
    let l = s.length;
    if (l === 0) return { nType: 7, type: "NON_STANDARD", reqSigs: 0, "script": s };
    if (A.segwit) {
      if ((l === 22) && (s[0] === 0))
        return { nType: 5, type: "P2WPKH", reqSigs: 1, addressHash: s.slice(2) };
      if ((l === 34) && (s[0] === 0))
        return { nType: 6, type: "P2WSH", reqSigs: null, addressHash: s.slice(2) };
    }

    if ((l === 25) && (s[0] === 0x76) && (s[1] === 0xa9) && (s[l - 2] === 0x88) && (s[l - 1] === 0xac))
      return { nType: 0, type: "P2PKH", reqSigs: 1, addressHash: s.slice(3, -2) };
    if ((l === 23) && (s[0] === 169) && (s[l - 1] === 135))
      return { nType: 1, type: "P2SH", reqSigs: null, addressHash: s.slice(2, -1) };
    if (((l === 67) || (l === 35)) && (s[l - 1] === 172))
      return { nType: 2, type: "PUBKEY", reqSigs: 1, addressHash: window.hash160(s.slice(1, -1)) };

    if (s[0] === O.OP_RETURN) {
      if (l === 1) return { nType: 3, type: "NULL_DATA", reqSigs: 0, "data": s.slice(1) };
      if ((s[1] < O.OP_PUSHDATA1) && (s[1] === l - 2))
        return { nType: 3, type: "NULL_DATA", reqSigs: 0, "data": s.slice(2) };
      if ((s[1] === O.OP_PUSHDATA1) && (l > 2) && (s[2] === l - 3) && (s[2] <= 80))
        return { nType: 3, type: "NULL_DATA", reqSigs: 0, "data": s.slice(3) };
      return { nType: 8, type: "NULL_DATA_NON_STANDARD", reqSigs: 0, "script": s }
    }

    if ((s[0] >= 81) && (s[0] <= 96) && (s[l - 1] === 174) && (s[l - 2] >= 81) && (s[l - 2] <= 96) && (s[l - 2] >= s[0])) {
      let c = 0, q = 1;
      while (l - q - 2 > 0) {
        if (s[q] < 0x4c) {
          q += s[q];
          c++;
        } else {
          q = 0;
          break;
        }
        q++;
      }
      if (c === s[l - 2] - 80)
        return { nType: 4, type: "MULTISIG", reqSigs: s[0] - 80, "pubKeys": c, "script": s }
    }

    let q = 0, m = 0, n = 0, last = 0, r = 0;
    while (l - q > 0) {
      if ((s[q] >= 81) && (s[q] <= 96)) {
        if (!n) n = s[q] - 80;
        else {
          if ((m === 0) || (m > n)) {
            n = s[q] - 80;
            m = 0;
          } else if (m === s[q] - 80) last = (last) ? 0 : 2;
        }
      } else if (s[q] < 0x4c) {
        q += s[q];
        m++;
        if (m > 16) {
          m = 0;
          n = 0;
        }
      } else if (s[q] === O.OP_PUSHDATA1) {
        if (s[q + 1] === undefined) break;
        q += 1 + s[q + 1];
      } else if (s[q] === O.OP_PUSHDATA2) {
        if (s[q + 1] === undefined) break;
        q += 2 + s.readIntLE(q, 2);
      } else if (s[q] === O.OP_PUSHDATA4) {
        if (s[q + 3] === undefined) break;
        q += 4 + s.readIntLE(q, 4);
      } else {
        if (s[q] === O.OP_CHECKSIG) r++;
        else if (s[q] === O.OP_CHECKSIGVERIFY) r++;
        else if ([O.OP_CHECKMULTISIG, O.OP_CHECKMULTISIGVERIFY].includes(s[q])) {
          if (last) r += n;
          else r += 20;
        }
        n = 0;
        m = 0;
      }
      if (last) last--;
      q++;
    }
    return { nType: 7, type: "NON_STANDARD", reqSigs: r, "script": s }
  };

  window.scriptToAddress = (s, A = {}) => {
    ARGS(A, { testnet: false });
    s = window.parseScript(s);
    if (s.addressHash !== undefined) {
      let wv = ((s.nType === 5) || (s.nType === 6)) ? 0 : null;
      let sh = ((s.nType === 1) || (s.nType === 6));
      return window.hashToAddress(s.addressHash, { testnet: A.testnet, scriptHash: sh, witnessVersion: wv })
    }
    return null;
  };

  /**
 *Decode script to ASM format or to human readable OPCODES string.
 *
 * :parameters:
 *   :s: script in bytes string or HEX encoded string format.
 * :param asm:  (optional) If set to ``true`` decode to ASM format, by default is ``false``.
 * :return: script in ASM format string or OPCODES string.
 */
  window.decodeScript = (s, A = {}) => {
    ARGS(A, { asm: false });
    s = getBuffer(s);
    let l = s.length, q = 0, result = [];
    if (l === 0) return '';
    try {
      while (l - q > 0) {
        if ((s[q] < 0x4c) && (s[q])) {
          if (A.asm) {
            result.push(`OP_PUSHBYTES[${s[q]}]`);
            result.push(s.slice(q + 1, q + 1 + s[q]).hex());
          } else result.push(`[${s[q]}]`);
          q += s[q] + 1;
          continue;
        }
        if (s[q] === O.OP_PUSHDATA1) {
          if (A.asm) {
            result.push(`OP_PUSHDATA1[${s[q + 1]}]`);
            result.push(s.slice(q + 2, q + 2 + s[q + 1]).hex());
          } else {
            result.push(RO[s[q]]);
            result.push(`[${s[q + 1]}]`);
          }
          q += 1 + s[q + 1] + 1;
          continue;
        }
        if (s[q] === O.OP_PUSHDATA2) {
          let w = s.readIntLE(q + 1, 2);
          if (A.asm) {
            result.push(`OP_PUSHDATA2[${w}]`);
            result.push(s.slice(q + 3, q + 3 + w).hex());
          } else {
            result.push(RO[s[q]]);
            result.push(`[${s[w]}]`);
          }
          q += w + 3;
          continue;
        }
        if (s[q] === O.OP_PUSHDATA4) {
          let w = s.readIntLE(q + 1, 4);
          if (A.asm) {
            result.push(`OP_PUSHDATA4[${w}]`);
            result.push(s.slice(q + 5, q + 5 + w).hex());
          } else {
            result.push(RO[s[q]]);
            result.push(`[${s[w]}]`);
          }
          q += w + 6;
          continue;
        }
        result.push(RO[s[q]]);
        q++;
      }
    } catch (e) {
      result.push("[SCRIPT_DECODE_FAILED]");
    }
    return result.join(' ');
  };

  /**
  * Delete OP_CODE or subscript from script.
  *
  * :parameters:
  *   :script: target script in bytes or HEX encoded string.
  *   :subScript: sub_script which is necessary to remove from target script in bytes or HEX encoded string.
  * :param hex: (optional) return HEX encoded string result flag, by default is ``false``.
  * :return: script in bytes or HEX encoded string corresponding to the format of target script.
  *
  */
  window.deleteFromScript = (script, subScript, A = {}) => {
    ARGS(A, { hex: false });
    if (subScript === undefined) return script;
    if (subScript.length === 0) return script;
    let s = getBuffer(script);
    let s2 = getBuffer(subScript);
    let l = s.length, ls = s2.length;
    let q = 0, k = 0, stack = [], result = [];
    while (l - q > 0) {
      if ((s[q] < 0x4c) && (s[q])) {
        stack.push(s[q] + 1);
        q += s[q] + 1;
      }
      else if (s[q] === O.OP_PUSHDATA1) {
        stack.push(1 + s[q + 1]);
        q += 1 + s[q + 1];
      }
      else if (s[q] === O.OP_PUSHDATA2) {
        let w = s.readIntLE(q, 2);
        stack.push(2 + w);
        q += 2 + w;
      }
      else if (s[q] === O.OP_PUSHDATA4) {
        let w = s.readIntLE(q, 4);
        stack.push(4 + w);
        q += 4 + w;
      }
      else {
        stack.push(1);
        q += 1;
      }

      if (q - k >= ls) {
        if (s.slice(k, q).slice(0, ls).equals(s2)) {
          if (q - k > ls) result.push(s.slice(k + ls, q));
          let t = 0;
          while (t !== q - k) t += stack.shift();
          k = q;
        }
        else {
          let t = stack.shift();
          result.push(s.slice(k, k + t));
          k += t;
        }
      }
    }

    if (s.slice(k, q).slice(0, ls).equals(s2)) {
      if (q - k > ls) result.push(s.slice(k + ls, q));
    }
    else result.push(s.slice(k, k + ls));

    let out = BC(result);
    return (A.hex) ? out.hex() : out;
  };

  /**
  * Encode script to hash HASH160 or SHA256 in dependency of the witness.
  *
  * :parameters:
  *   :s: script in bytes or HEX encoded string.
  * :param witness: (optional) If set to ``true`` return SHA256 hash for P2WSH, by default is ``false``.
  * :param hex: (optional) return HEX encoded string result flag, by default is ``true``.
  * :return: hash HASH160 or SHA256 of script in bytes or HEX encoded.
  */
  window.scriptToHash = (s, A = {}) => {
    ARGS(A, { witness: false, hex: true });
    return (A.witness) ? window.sha256(s, A) : window.hash160(s, A)
  };

  window.opPushData = (s) => {
    if (s.length <= 0x4b) return BC([BF([s.length]), s]);
    if (s.length <= 0xff) return BC([BF([O.OP_PUSHDATA1, s.length]), s]);
    if (s.length <= 0xffff) return BC([BF([O.OP_PUSHDATA2].concat(window.intToBytes(s.length, 2, 'little'))), s]);
    return BC([BF([O.OP_PUSHDATA4].concat(window.intToBytes(s.length, 4, 'little'))), s]);
  };

  window.readOpCode = (s) => {
    let b = s.read(1);
    if (!b.length) return [null, null];
    if (b[0] <= 0x4b) return [b, s.read(b[0])];
    if (b[0] === O.OP_PUSHDATA1) return [b, s.read(s.read(1)[0])];
    if (b[0] === O.OP_PUSHDATA2) return [b, s.read(s.read(2).readIntLE(0, 2))];
    if (b[0] === O.OP_PUSHDATA4) return [b, s.read(s.read(4).readIntLE(0, 4))];
    return [b, null]
  };

  window.getMultiSigPublicKeys = (s) => {
    let keys = [];
    let r = window.readOpCode(s);
    while (r[0] !== null) {
      r = window.readOpCode(s);
      if (r[1] !== null) keys.push(r[1]);
    }
    return keys;
  };

  window.signMessage = (m, privateKey, A = {}) => {
    ARGS(A, { encoding: 'hex|utf8', hex: false });
    m = getBuffer(m, A.encoding);
    if (window.isString(privateKey)) {
      if (window.isHex(privateKey)) privateKey = BF(privateKey, 'hex');
      else if (window.isWifValid(privateKey)) privateKey = window.wifToPrivateKey(privateKey, { hex: false });
      else throw new Error("private key invalid");
    }
    else if (!Buffer.isBuffer(privateKey)) privateKey = BF(privateKey);
    if (privateKey.length !== 32) throw new Error("private key length invalid");
    if (m.length !== 32) throw new Error("message length invalid");

    let mP, pP, sP, len, signature, sDp, r = 0, recId;
    try {
      mP = malloc(32);
      pP = malloc(32);
      sP = malloc(65);
      sDp = malloc(72);
      len = malloc(1);
      CM.HEAPU8.set(m, mP);
      CM.HEAPU8.set(privateKey, pP);
      r = CM._secp256k1_ecdsa_sign_recoverable(window.secp256k1PrecompContextSign, sP, mP, pP, null, null);
      if (r) {
        recId = getValue(sP + 64, 'i8');
        r = CM._secp256k1_ecdsa_signature_serialize_der(window.secp256k1PrecompContextSign, sDp, len, sP);
        if (r) {
          let l = getValue(len, 'i8');
          signature = new Buffer.alloc(l);
          for (let i = 0; i < l; i++) signature[i] = getValue(sDp + i, 'i8');
        }
      }


    } finally {
      free(mP);
      free(pP);
      free(sP);
      free(sDp);
      free(len);
    }

    if (r) {
      return {
        signature: (A.hex) ? signature.hex() : signature,
        recId: recId
      }
    }
    return null;
  };

  /**
  * Verify signature for message and given public key
  *
  * :parameters:
  *   :s: signature in bytes or HEX encoded string.
  *   :p: public key in bytes or HEX encoded string.
  *   :m: message in bytes or HEX encoded string.
  * :return: boolean.
  */
  window.verifySignature = (s, p, m) => {
    s = getBuffer(s);
    p = getBuffer(p);
    m = getBuffer(m);
    let mP, pP, sP, sCp, pCp, r = 0;
    try {
      mP = malloc(m.length);
      pP = malloc(p.length);
      sP = malloc(s.length);
      sCp = malloc(64);
      pCp = malloc(65);
      CM.HEAPU8.set(m, mP);
      CM.HEAPU8.set(p, pP);
      CM.HEAPU8.set(s, sP);
      r = CM._secp256k1_ecdsa_signature_parse_der(window.secp256k1PrecompContextSign, sCp, sP, s.length);
      if (r) {
        r = CM._secp256k1_ec_pubkey_parse(window.secp256k1PrecompContextVerify, pCp, pP, p.length);
        if (r) {
          r = CM._secp256k1_ecdsa_verify(window.secp256k1PrecompContextVerify, sCp, mP, pCp);
        }
      }
    }
    finally {
      free(mP);
      free(pP);
      free(sP);
      free(sCp);
    }
    return Boolean(r);
  };

  window.publicKeyRecovery = (s, m, recId, A = {}) => {
    ARGS(A, { compressed: true, hex: true, der: true });
    s = getBuffer(s);
    m = getBuffer(m);

    let mP, sCp, sP, sCcP, sCcRp, lP, sPub, pub, out, r = 0, len, flag;
    try {
      mP = malloc(m.length);
      sP = malloc(s.length);
      sCp = malloc(64);
      sCcP = malloc(64);
      sCcRp = malloc(65);
      pub = malloc(65);
      sPub = malloc(65);
      lP = malloc(1);
      CM.HEAPU8.set(m, mP);
      CM.HEAPU8.set(s, sP);
      if (A.der) {
        r = CM._secp256k1_ecdsa_signature_parse_der(window.secp256k1PrecompContextSign, sCp, sP, s.length);
        if (r) r = CM._secp256k1_ecdsa_signature_serialize_compact(window.secp256k1PrecompContextSign, sCcP, sCp);
        if (r) r = CM._secp256k1_ecdsa_recoverable_signature_parse_compact(window.secp256k1PrecompContextSign, sCcRp, sCcP, recId);
      }
      else
        r = CM._secp256k1_ecdsa_recoverable_signature_parse_compact(window.secp256k1PrecompContextSign, sCcRp, sP, recId);

      if (r) r = CM._secp256k1_ecdsa_recover(window.secp256k1PrecompContextVerify, pub, sCcRp, mP);
      if (r) {
        if (A.compressed) {
          len = 33;
          flag = window.SECP256K1_EC_COMPRESSED;
        } else {
          len = 65;
          flag = window.SECP256K1_EC_UNCOMPRESSED;
        }
        CM.HEAP8.set([len], lP);
        r = CM._secp256k1_ec_pubkey_serialize(window.secp256k1PrecompContextVerify, sPub, lP, pub, flag);
      }
      if (r) {
        out = new Buffer.alloc(len);
        for (let i = 0; i < len; i++) out[i] = getValue(sPub + i, 'i8');
      }
    } finally {
      free(mP);
      free(sP);
      free(sCp);
      free(sCcP);
      free(sCcRp);
      free(pub);
      free(sPub);
      free(lP);
    }

    if (r) return (A.hex) ? out.hex() : out;
    return null;
  };

  /**
  * Check is valid signature encoded in DER format
  *
  * :parameters:
  *   :s: signature in bytes or HEX encoded string.
  * :return:  boolean.
  */

  window.isValidSignatureEncoding = (s) => {
    // # Format: 0x30 [total-length] 0x02 [R-length] [R] 0x02 [S-length] [S] [sighash]
    // # * total-length: 1-byte length descriptor of everything that follows,
    //     #   excluding the sighash byte.
    //     # * R-length: 1-byte length descriptor of the R value that follows.
    //     # * R: arbitrary-length big-endian encoded R value. It must use the shortest
    // #   possible encoding for a positive integers (which means no null bytes at
    // #   the start, except a single one when the next byte has its highest bit set).
    // # * S-length: 1-byte length descriptor of the S value that follows.
    //     # * S: arbitrary-length big-endian encoded S value. The same rules apply.
    //     # * sighash: 1-byte value indicating what data is hashed (not part of the DER
    // #   signature)
    s = getBuffer(s);
    let l = s.length;
    if (((l < 9) || (l > 73)) || (s[0] !== 0x30) || (s[1] !== l - 3)) return false;
    let lR = s[3];
    if (5 + lR >= l) return false;
    let lS = s[5 + lR];
    if (((lR + lS + 7) !== l) || (s[2] !== 0x02) || (lR === 0) || (s[4] === 0x80)) return false;
    if (((lR > 1) && (s[4] === 0) && (!(s[5] & 0x80))) || (s[lR + 4] !== 0x02) || (lS === 0) || (s[lR + 6] & 0x80)) return false;
    return !((lS > 1) && (s[lR + 6] === 0) && (!(s[lR + 7] & 0x80)));

  };

  window.parseSignature = function (s, A = {}) {
    // # Format: 0x30 [total-length] 0x02 [R-length] [R] 0x02 [S-length] [S] [sighash]
    // # * total-length: 1-byte length descriptor of everything that follows,
    //     #   excluding the sighash byte.
    //     # * R-length: 1-byte length descriptor of the R value that follows.
    //     # * R: arbitrary-length big-endian encoded R value. It must use the shortest
    // #   possible encoding for a positive integers (which means no null bytes at
    // #   the start, except a single one when the next byte has its highest bit set).
    // # * S-length: 1-byte length descriptor of the S value that follows.
    //     # * S: arbitrary-length big-endian encoded S value. The same rules apply.
    //     # * sighash: 1-byte value indicating what data is hashed (not part of the DER
    // #   signature)
    ARGS(A, { hex: false });
    s = getBuffer(s);
    s.length;
    if (!this.isValidSignatureEncoding(s)) throw new Error('invalid signature');
    let lR = s[3];
    let r = s.slice(5, 4 + lR);
    let lS = s[5 + lR];
    s = s.slice(lR + 6, lS);
    return [(A.hex) ? r.hex() : r, (A.hex) ? s.hex() : s];
  };

  window.bitcoinMessage = function (msg) {
    return window.doubleSha256(BC([
      window.Buffer.from('\x18Bitcoin Signed Message:\n', 'utf8'),
      BF(window.intToVarInt(msg.length)),
      window.Buffer.from(msg, 'utf8')]), { hex: false });
  };


  /**
  * Sign message
  *
  * :parameters:
  *   :m: message in bytes or HEX encoded string.
  *   :privateKey: private key (bytes, hex encoded string or WIF format)
  * :param base64: (optional) If set to ``true`` return key signature BASE64 format, by default is ``true``.
  * :return: DER or BASE64 encoded signature in bytes.
  */
  window.signBitcoinMessage = (m, privateKey, A = {}) => {
    ARGS(A, { base64: true });
    m = window.bitcoinMessage(m);
    let compressed = 1;
    if (window.isString(privateKey)) {
      if (window.isHex(privateKey)) privateKey = BF(privateKey, 'hex');
      else if (window.isWifValid(privateKey)) {
        if ((privateKey[0] === window.MAINNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX) ||
          (privateKey[0] === window.TESTNET_PRIVATE_KEY_UNCOMPRESSED_PREFIX)) compressed = 0;
        privateKey = window.wifToPrivateKey(privateKey, { hex: false });

      }
      else throw new Error("private key invalid");
    }
    else if (!Buffer.isBuffer(privateKey)) privateKey = BF(privateKey);
    if (privateKey.length !== 32) throw new Error("private key length invalid");

    let mP, pP, sP, signature, sDp, r = 0;
    try {
      mP = malloc(32);
      pP = malloc(32);
      sP = malloc(65);
      sDp = malloc(65);
      CM.HEAPU8.set(m, mP);
      CM.HEAPU8.set(privateKey, pP);
      r = CM._secp256k1_ecdsa_sign_recoverable(window.secp256k1PrecompContextSign, sP, mP, pP, null, null);
      if (r) {
        r = CM._secp256k1_ecdsa_recoverable_signature_serialize_compact(window.secp256k1PrecompContextSign, sDp + 1,
          sDp, sP);
        if (r) {
          signature = new Buffer.alloc(65);
          for (let i = 0; i < 65; i++) signature[i] = getValue(sDp + i, 'i8');
          signature[0] = signature[0] + 27 + 4 * compressed;
        }
      }
    } finally {
      free(mP);
      free(pP);
      free(sP);
      free(sDp);
    }

    if (r) return (A.base64) ? signature.toString('base64') : signature;
    return null;
  };

  window.bitcoinSignedMessageAddresses = (m, s, A = {}) => {
    ARGS(A, { testnet: false });
    if (window.isString(s)) s = BF(s, 'base64');

    let recId = s.readInt8([0]);
    if ((recId < 27) || (recId >= 35)) return [];
    let compressed = true;
    if (recId >= 31) recId -= 4;
    else compressed = false;
    recId -= 27;

    m = window.bitcoinMessage(m);
    let pub = window.publicKeyRecovery(s.slice(1), m, recId, { compressed: compressed, hex: true, der: false });
    if (pub !== null) {
      if (compressed)
        return [window.publicKeyToAddress(pub, { testnet: A.testnet, p2sh_p2wpkh: false, witnessVersion: 0 }),
        window.publicKeyToAddress(pub, { testnet: A.testnet, p2sh_p2wpkh: false, witnessVersion: null }),
        window.publicKeyToAddress(pub, { testnet: A.testnet, p2sh_p2wpkh: true, witnessVersion: 0 })];
      else return [window.publicKeyToAddress(pub, { testnet: A.testnet, p2sh_p2wpkh: false, witnessVersion: null })]
    }
    return [];
  };

  window.verifyBitcoinMessage = (m, s, a, A = {}) => {
    ARGS(A, { testnet: false });
    let addresses = window.bitcoinSignedMessageAddresses(m, s, A);
    return addresses.includes(a);
  };

}

function shamirSecret(window) {
  let BA = window.Buffer.alloc;
  let BF = window.Buffer.from;
  let BC = window.Buffer.concat;


  window.__precompute_GF256_expLog = (S) => {
    let exp = BA(255, 0);
    let log = BA(256, 0);
    let poly = 1;
    for (let i = 0; i < 255; i++) {
      exp[i] = poly;
      log[poly] = i;
      // Multiply poly by the polynomial x + 1.
      poly = (poly << 1) ^ poly;
      // Reduce poly by x^8 + x^4 + x^3 + x + 1
      if (poly & 0x100) poly ^= 0x11b;
    }
    S.GF256_EXP_TABLE = exp;
    S.GF256_LOG_TABLE = log;
  };

  window.__GF256_mul = (a, b) => {
    if ((a === 0) || (b === 0)) return 0;
    return window.GF256_EXP_TABLE[window.__mod(window.GF256_LOG_TABLE[a] + window.GF256_LOG_TABLE[b], 255)];
  };

  window.__GF256_pow = (a, b) => {
    if (b === 0) return 1;
    if (a === 0) return 0;
    let c = a;
    for (let i = 0; i < b - 1; i++) c = window.__GF256_mul(c, a);
    return c;
  };

  window.__mod = (a, b) => ((a % b) + b) % b;

  window.__GF256_add = (a, b) => a ^ b;

  window.__GF256_sub = (a, b) => a ^ b;

  window.__GF256_inverse = (a) => {
    if (a === 0) throw new Error("Zero division");
    return window.GF256_EXP_TABLE[window.__mod(-1 * window.GF256_LOG_TABLE[a], 255)];
  };

  window.__GF256_div = (a, b) => {
    if (b === 0) throw new Error("Zero division");
    if (a === 0) return 0;
    let r = window.GF256_EXP_TABLE[window.__mod(window.GF256_LOG_TABLE[a] - window.GF256_LOG_TABLE[b], 255)];
    // let r = S.__GF256_mul(a, S.__GF256_inverse(b));
    if (a !== window.__GF256_mul(r, b)) throw new Error("failed");
    return r;
  };

  window.__shamirFn = (x, q) => {
    let r = 0;
    for (let a of q) r = window.__GF256_add(r, window.__GF256_mul(a, window.__GF256_pow(x, q.indexOf(a))));
    return r;
  };

  window.__shamirInterpolation = (points) => {
    let k = points.length;
    if (k < 2) throw new Error("Minimum 2 points required");
    points.sort((a, b) => a[0] - b[0]);
    let z = new Set();
    for (let i of points) z.add(i[0]);
    if (z.size !== points.length) throw new Error("Unique points required");
    let p_x = 0;
    for (let j = 0; j < k; j++) {
      let p_j_x = 1;
      for (let m = 0; m < k; m++) {
        if (m === j) continue;
        // let a = S.__GF256_sub(x, points[m][0]);
        let a = points[m][0];
        // let b = S.__GF256_sub(points[j][0], points[m][0]);
        let b = window.__GF256_add(points[j][0], points[m][0]);
        let c = window.__GF256_div(a, b);
        p_j_x = window.__GF256_mul(p_j_x, c);
      }
      p_j_x = window.__GF256_mul(points[j][1], p_j_x);
      p_x = window.__GF256_add(p_x, p_j_x);
    }
    return p_x;
  };


  window.__split_secret = (threshold, total, secret, indexBits = 8) => {
    if (threshold > 255) throw new Error("threshold limit 255");
    if (total > 255) throw new Error("total limit 255");
    let index_mask = 2 ** indexBits - 1;
    if (total > index_mask) throw new Error("index bits is to low");
    if (threshold > total) throw new Error("invalid threshold");
    let shares = {};
    let sharesIndexes = [];

    let e = window.generateEntropy({ hex: false });
    let ePointer = 0;
    let i = 0;
    let index;

    // generate random indexes (x coordinate)
    do {
      if (ePointer >= e.length) {
        // get more 32 bytes entropy
        e = window.generateEntropy({ hex: false });
        ePointer = 0;
      }

      index = e[ePointer] & index_mask;
      if ((shares[index] === undefined) && (index !== 0)) {
        i++;
        shares[index] = BF([]);
        sharesIndexes.push(index);
      }

      ePointer++;
    } while (i !== total);


    e = window.generateEntropy({ hex: false });
    ePointer = 0;

    let w;
    for (let b = 0; b < secret.length; b++) {
      let q = [secret[b]];

      for (let i = 0; i < threshold - 1; i++) {
        do {
          if (ePointer >= e.length) {
            ePointer = 0;
            e = window.generateEntropy({ hex: false });
          }
          w = e[ePointer++];
        } while (q.includes(w));
        q.push(w);
      }

      for (let i of sharesIndexes)
        shares[i] = BC([shares[i], BF([window.__shamirFn(i, q)])]);

    }
    return shares;
  };

  window.__restore_secret = (shares) => {
    let secret = BF([]);
    let shareLength = null;

    for (let i in shares) {
      i = parseInt(i);
      if ((i < 1) || (i > 255)) throw new Error("Invalid share index " + i);
      if (shareLength === null) shareLength = shares[i].length;
      if ((shareLength !== shares[i].length) || (shareLength === 0)) throw new Error("Invalid shares");
    }

    for (let i = 0; i < shareLength; i++) {
      let points = [];
      for (let z in shares) {
        z = parseInt(z);
        points.push([z, shares[z][i]]);
      }
      secret = BC([secret, BF([window.__shamirInterpolation(points)])]);
    }
    return secret;
  };

  window.__precompute_GF256_expLog(window);
}

function tools(window) {
  window.Buffer = Buffer$1;
  window.isBuffer = window.Buffer.isBuffer;
  window.BN = BN;
  window.__nodeCrypto = false;
  try {
    // @TODO: apply pollyfill for browser
    window.__nodeCrypto = require('crypto');
  } catch (e) {
  }

  window.Buffer.prototype.seek = function (n) {
    this.__offset = (n > this.length) ? this.length : n;
  };

  window.Buffer.prototype.tell = function () {
    this.__offset;
  };

  window.Buffer.prototype.read = function (n) {
    if (this.__offset === undefined) this.__offset = 0;
    if (this.__offset === this.length) return window.Buffer.from([]);
    let m = this.__offset + n;
    if (m > this.length) m = this.length;
    let r = this.slice(this.__offset, m);
    this.__offset = m;
    return r;
  };

  window.Buffer.prototype.readVarInt = function () {
    if (this.__offset === undefined) this.__offset = 0;
    if (this.__offset === this.length) return window.Buffer.from([]);
    let l = this[this.__offset];
    if (l < 253) l = 1;
    else if (l === 253) l = 3;
    else if (l === 254) l = 5;
    else if (l === 255) l = 9;
    return this.read(l);
  };

  window.Buffer.prototype.readInt = function (n, byte_order = 'little') {
    if (this.__offset === undefined) this.__offset = 0;
    if (this.__offset === this.length) return 0;
    if ((this.__offset + n) > this.length) n = this.length - this.__offset;
    let r;
    if (byte_order === 'little') r = this.readUIntLE(this.__offset, n);
    else r = this.readUIntBE(this.__offset, n);
    this.__offset += n;
    return r;
  };

  window.Buffer.prototype.hex = function () {
    return this.toString('hex');
  };

  window.getWindow = () => {
    if (typeof window !== "undefined") return window;
    if (typeof global !== "undefined") return global;
    if (typeof self !== "undefined") return self;
    return {};
  };

  window.readVarInt = (s) => {
    let l = s[s.__offset];
    if (l < 253) l = 1;
    else if (l === 253) l = 3;
    else if (l === 254) l = 5;
    else if (l === 255) l = 9;
    return s.read(l);
  };


  window.BNZerro = new window.BN(0);

  window.isHex = s => Boolean(/^[0-9a-fA-F]+$/.test(s) && !(s.length % 2));

  window.getBuffer = function (m, encoding = 'hex') {
    if (window.isBuffer(m)) {
      if (m.read === undefined) return window.Buffer.from(m);
      return m;
    }
    if (window.isString(m)) {
      if (m.length === 0) return window.Buffer(0);
      encoding = encoding.split('|');
      for (let e of encoding) {
        if (e === 'hex') {
          if (window.isHex(m)) return window.Buffer.from(m, e);
        } else if (e === 'utf8') return window.Buffer.from(m, e);
      }
      throw new Error(encoding + ' encoding required :' + encoding);
    }
    return window.Buffer.from(m);
  };

  window.isString = function (value) {
    return typeof value === 'string' || value instanceof String;
  };

  window.defArgs = function (n, v) {
    if (!(n instanceof Object) && (n !== undefined)) throw new Error("Invalid named arguments object");
    for (let k in v) if (n[k] === undefined) n[k] = v[k];
  };


  /**
  * Convert bytes to string.
  *
  * :parameters:
  *    :bytes: bytes
  * :return: string.
  */
  window.bytesToString = function (bytes) {
    return bytes.map(function (x) {
      return String.fromCharCode(x)
    }).join('')
  };

  /**
  * Convert HEX string to bytes.
  *
  * :parameters:
  *    :hex: HEX string
  * :return: bytes.
  */
  window.hexToBytes = (hex) => {
    if (hex.length % 2 === 1) throw new Error("hexToBytes can't have a string with an odd number of characters.");
    if (hex.indexOf('0x') === 0) hex = hex.slice(2);
    return hex.match(/../g).map(function (x) {
      return parseInt(x, 16)
    })
  };

  /**
  * Convert string to bytes.
  *
  * :parameters:
  *    :str: string
  * :return: bytes.
  */
  window.stringToBytes = function (str) {
    return str.split('').map(function (x) {
      return x.charCodeAt(0)
    })
  };

  /**
  * Convert integer to bytes.
  *
  * :parameters:
  *   :x: integer.
  *   :n: bytes count.
  *   :byte_order: (optional) byte order 'big' or 'little', by default is 'little'.
  * :return: bytes.
  */
  window.intToBytes = function (x, n, byte_order = "little") {
    let bytes = [];
    let i = n;
    if (n === undefined) throw new Error('bytes count required');
    if ((byte_order !== "big") && (byte_order !== "little")) throw new Error('invalid byte order');
    let b = (byte_order === "big");
    if (n <= 4)
      do {
        (b) ? bytes.unshift(x & (255)) : bytes.push(x & (255));
        x = x >> 8;
      } while (--i);
    else {
      x = new window.BN(x);
      bytes = x.toArrayLike(Array, (b) ? 'be' : 'le', n);
    }
    return bytes;
  };
  /**
  * Convert integer to variable integer
  *
  * :parameters:
  *    :i: integer.
  * :return: bytes.
  */
  window.intToVarInt = function (i) {
    let r;
    if (i instanceof window.BN) {
      if (i.lt(0xfd)) r = i.toArrayLike(Array, 'le', 1);
      else if (i.lt(0xffff)) r = [0xfd].concat(i.toArrayLike(Array, 'le', 2));
      else if (i.lt(0xffffffff)) r = [0xfe].concat(i.toArrayLike(Array, 'le', 4));
      else r = [0xff].concat(i.toArrayLike(Array, 'le', 8));
      return r;
    } else if (!isNaN(i)) {
      if (i < 0xfd) r = [i];
      else if (i < 0xffff) r = [0xfd].concat(window.intToBytes(i, 2, 'little'));
      else if (i < 0xffffffff) r = [0xfe].concat(window.intToBytes(i, 4, 'little'));
      else r = [0xff].concat(window.intToBytes(i, 8, 'little'));
      return r;
    } else {
      throw new Error('invalid argument type', i);
    }
  };

  /**
  * Convert variable integer to integer
  *
  * :parameters:
  *    :s: bytes variable integer.
  *    :bn: (optional) BigNum flag, by default is ``false``
  * :return: integer.
  */
  window.varIntToInt = function (s, bn = false) {
    let r;
    if (s[0] < 0xfd) r = new window.BN(s[0]);
    else if (s[0] < 0xffff) r = new window.BN(s.slice(1, 3), 'le');
    else if (s[0] < 0xffffffff) r = new window.BN(s.slice(1, 4), 'le');
    else r = new window.BN(s.slice(1, 8), 'le');
    if (bn) return r;
    return r.toNumber();
  };

  /**
 * Get variable integer length in bytes from integer value
 *
 * :parameters:
 *    :b: integer
 * :return: integer.
 */
  window.varIntLen = (b) => (b[0] < 0xfd) ? 1 : (b[0] < 0xffff) ? 2 : (b[0] < 0xffffffff) ? 4 : 8;


  /**
  * Encode raw transaction hash to HEX string with bytes order change
  *
  * :parameters:
  *   :raw_hash: transaction hash in bytes string.
  * :return:  HEX encoded string.
  */
  window.rh2s = (s) => window.Buffer.from(s).reverse().hex();

  /**
  * Decode HEX  transaction hash to bytes with byte order change
  *
  * :parameters:
  *   :hash_string: HEX encoded string.
  * :return:  bytes string.
  */
  window.s2rh = (s) => window.Buffer.from(s, 'hex').reverse();

}

// ESM convert
// build
const OPCODE = {
  OP_FALSE: 0x00,
  OP_0: 0x00,
  OP_PUSHDATA1: 0x4c,
  OP_PUSHDATA2: 0x4d,
  OP_PUSHDATA4: 0x4e,
  OP_1NEGATE: 0x4f,
  OP_RESERVED: 0x50,
  OP_TRUE: 0x51,
  OP_1: 0x51,
  OP_2: 0x52,
  OP_3: 0x53,
  OP_4: 0x54,
  OP_5: 0x55,
  OP_6: 0x56,
  OP_7: 0x57,
  OP_8: 0x58,
  OP_9: 0x59,
  OP_10: 0x5a,
  OP_11: 0x5b,
  OP_12: 0x5c,
  OP_13: 0x5d,
  OP_14: 0x5e,
  OP_15: 0x5f,
  OP_16: 0x60,
  // control
  OP_NOP: 0x61,
  OP_VER: 0x62,
  OP_IF: 0x63,
  OP_NOTIF: 0x64,
  OP_VERIF: 0x65,
  OP_ELSE: 0x67,
  OP_ENDIF: 0x68,
  OP_VERIFY: 0x69,
  OP_RETURN: 0x6a,
  // stack
  OP_TOALTSTACK: 0x6b,
  OP_FROMALTSTACK: 0x6c,
  OP_2DROP: 0x6d,
  OP_2DUP: 0x6e,
  OP_3DUP: 0x6f,
  OP_2OVER: 0x70,
  OP_2ROT: 0x71,
  OP_2SWAP: 0x72,
  OP_IFDUP: 0x73,
  OP_DEPTH: 0x74,
  OP_DROP: 0x75,
  OP_DUP: 0x76,
  OP_NIP: 0x77,
  OP_OVER: 0x78,
  OP_PICK: 0x79,
  OP_ROLL: 0x7a,
  OP_ROT: 0x7b,
  OP_SWA: 0x7c,
  OP_TUCK: 0x7d,
  // splice
  OP_CAT: 0x7e,
  OP_SUBSTR: 0x7f,
  OP_LEFT: 0x80,
  OP_RIGHT: 0x81,
  OP_SIZE: 0x82,
  // bit operations
  OP_INVERT: 0x83,
  OP_AND: 0x84,
  OP_OR: 0x85,
  OP_XOR: 0x86,
  OP_EQUAL: 0x87,
  OP_EQUALVERIFY: 0x88,
  OP_RESERVED1: 0x89,
  OP_RESERVED2: 0x8a,
  // math
  OP_1ADD: 0x8b,
  OP_1SUB: 0x8c,
  OP_2MUL: 0x8d,
  OP_2DIV: 0x8e,
  OP_NEGATE: 0x8f,
  OP_ABS: 0x90,
  OP_NOT: 0x91,
  OP_NOTEQUAL: 0x92,
  OP_ADD: 0x93,
  OP_SUB: 0x94,
  OP_MUL: 0x95,
  OP_DIV: 0x96,
  OP_MOD: 0x97,
  OP_LSHIFT: 0x98,
  OP_RSHIFT: 0x99,

  OP_BOOLAND: 0x9a,
  OP_BOOLOR: 0x9b,
  OP_NUMEQUAL: 0x9c,
  OP_NUMEQUALVERIFY: 0x9d,
  OP_NUMNOTEQUAL: 0x9e,
  OP_LESSTHAN: 0x9f,
  OP_GREATERTHAN: 0xa0,
  OP_LESSTHANOREQUAL: 0xa1,
  OP_GREATERTHANOREQUAL: 0xa2,
  OP_MIN: 0xa3,
  OP_MAX: 0xa4,

  OP_WITHIN: 0xa5,

  // crypto
  OP_RIPEMD160: 0xa6,
  OP_SHA1: 0xa7,
  OP_SHA256: 0xa8,
  OP_HASH160: 0xa9,
  OP_HASH256: 0xaa,
  OP_CODESEPARATOR: 0xab,
  OP_CHECKSIG: 0xac,
  OP_CHECKSIGVERIFY: 0xad,
  OP_CHECKMULTISIG: 0xae,
  OP_CHECKMULTISIGVERIFY: 0xaf,

  // expansion
  OP_NOP1: 0xb0,
  OP_CHECKLOCKTIMEVERIFY: 0xb1,
  OP_CHECKSEQUENCEVERIFY: 0xb2,
  OP_NOP4: 0xb3,
  OP_NOP5: 0xb4,
  OP_NOP6: 0xb5,
  OP_NOP7: 0xb6,
  OP_NOP8: 0xb7,
  OP_NOP9: 0xb8,
  OP_NOP10: 0xb9,

  // template matching params

  OP_SMALLINTEGER: 0xfa,
  OP_PUBKEYS: 0xfb,
  OP_PUBKEYHASH: 0xfd,
  OP_PUBKEY: 0xfe,
  OP_INVALIDOPCODE: 0xff
};
let RAW_OPCODE = {};
for (let i in OPCODE) RAW_OPCODE[OPCODE[i]] = i;

function opcodes(window) {
  window.OPCODE = OPCODE;
  window.RAW_OPCODE = RAW_OPCODE;
}//
// RAW_OPCODE = dict((OPCODE[i], i) for i in OPCODE)
// BYTE_OPCODE = dict((i, bytes([OPCODE[i]])) for i in OPCODE)
// HEX_OPCODE = dict((i, bytes([OPCODE[i]]).hex()) for i in OPCODE)
// for i in range(256):
// if i not in RAW_OPCODE:
// RAW_OPCODE[i]="OP_UNKNOWN"
//
// OP_FALSE = BYTE_OPCODE["OP_FALSE"]
// OP_0 = BYTE_OPCODE["OP_0"]
// OP_PUSHDATA1 = BYTE_OPCODE["OP_PUSHDATA1"]
// OP_PUSHDATA2 = BYTE_OPCODE["OP_PUSHDATA2"]
// OP_PUSHDATA4 = BYTE_OPCODE["OP_PUSHDATA4"]
// OP_1NEGATE = BYTE_OPCODE["OP_1NEGATE"]
// OP_RESERVED = BYTE_OPCODE["OP_RESERVED"]
// OP_1 = BYTE_OPCODE["OP_1"]
// OP_TRUE = BYTE_OPCODE["OP_TRUE"]
// OP_2 = BYTE_OPCODE["OP_2"]
// OP_3 = BYTE_OPCODE["OP_3"]
// OP_4 = BYTE_OPCODE["OP_4"]
// OP_5 = BYTE_OPCODE["OP_5"]
// OP_6 = BYTE_OPCODE["OP_6"]
// OP_7 = BYTE_OPCODE["OP_7"]
// OP_8 = BYTE_OPCODE["OP_8"]
// OP_9 = BYTE_OPCODE["OP_9"]
// OP_10 = BYTE_OPCODE["OP_10"]
// OP_11 = BYTE_OPCODE["OP_11"]
// OP_12 = BYTE_OPCODE["OP_12"]
// OP_13 = BYTE_OPCODE["OP_13"]
// OP_14 = BYTE_OPCODE["OP_14"]
// OP_15 = BYTE_OPCODE["OP_15"]
// OP_16 = BYTE_OPCODE["OP_16"]
//
// # control
//
// OP_NOP = BYTE_OPCODE["OP_NOP"]
// OP_VER = BYTE_OPCODE["OP_VER"]
// OP_IF = BYTE_OPCODE["OP_IF"]
// OP_NOTIF = BYTE_OPCODE["OP_NOTIF"]
// OP_VERIF = BYTE_OPCODE["OP_VERIF"]
// OP_ELSE = BYTE_OPCODE["OP_ELSE"]
// OP_ENDIF = BYTE_OPCODE["OP_ENDIF"]
// OP_VERIFY = BYTE_OPCODE["OP_VERIFY"]
// OP_RETURN = BYTE_OPCODE["OP_RETURN"]
//
// # stack
//
// OP_TOALTSTACK = BYTE_OPCODE["OP_TOALTSTACK"]
// OP_FROMALTSTACK = BYTE_OPCODE["OP_FROMALTSTACK"]
// OP_2DROP = BYTE_OPCODE["OP_2DROP"]
// OP_2DUP = BYTE_OPCODE["OP_2DUP"]
// OP_3DUP = BYTE_OPCODE["OP_3DUP"]
// OP_2OVER = BYTE_OPCODE["OP_2OVER"]
// OP_2ROT = BYTE_OPCODE["OP_2ROT"]
// OP_2SWAP = BYTE_OPCODE["OP_2SWAP"]
// OP_IFDUP = BYTE_OPCODE["OP_IFDUP"]
// OP_DEPTH = BYTE_OPCODE["OP_DEPTH"]
// OP_DROP = BYTE_OPCODE["OP_DROP"]
// OP_DUP = BYTE_OPCODE["OP_DUP"]
// OP_NIP = BYTE_OPCODE["OP_NIP"]
// OP_OVER = BYTE_OPCODE["OP_OVER"]
// OP_PICK = BYTE_OPCODE["OP_PICK"]
// OP_ROLL = BYTE_OPCODE["OP_ROLL"]
// OP_ROT = BYTE_OPCODE["OP_ROT"]
// OP_SWAP = BYTE_OPCODE["OP_SWAP"]
// OP_TUCK = BYTE_OPCODE["OP_TUCK"]
//
// # splice
//
// OP_CAT = BYTE_OPCODE["OP_CAT"]
// OP_SUBSTR = BYTE_OPCODE["OP_SUBSTR"]
// OP_LEFT = BYTE_OPCODE["OP_LEFT"]
// OP_RIGHT = BYTE_OPCODE["OP_RIGHT"]
// OP_SIZE = BYTE_OPCODE["OP_SIZE"]
//
// # bit operations
//
// OP_INVERT = BYTE_OPCODE["OP_INVERT"]
// OP_AND = BYTE_OPCODE["OP_AND"]
// OP_OR = BYTE_OPCODE["OP_OR"]
// OP_XOR = BYTE_OPCODE["OP_XOR"]
// OP_EQUAL = BYTE_OPCODE["OP_EQUAL"]
// OP_EQUALVERIFY = BYTE_OPCODE["OP_EQUALVERIFY"]
// OP_RESERVED1 = BYTE_OPCODE["OP_RESERVED1"]
// OP_RESERVED2 = BYTE_OPCODE["OP_RESERVED2"]
//
// # math
//
// OP_1ADD = BYTE_OPCODE["OP_1ADD"]
// OP_1SUB = BYTE_OPCODE["OP_1SUB"]
// OP_2MUL = BYTE_OPCODE["OP_2MUL"]
// OP_2DIV = BYTE_OPCODE["OP_2DIV"]
// OP_NEGATE = BYTE_OPCODE["OP_NEGATE"]
// OP_ABS = BYTE_OPCODE["OP_ABS"]
// OP_NOT = BYTE_OPCODE["OP_NOT"]
// OP_0NOTEQUAL = BYTE_OPCODE["OP_0NOTEQUAL"]
//
// OP_ADD = BYTE_OPCODE["OP_ADD"]
// OP_SUB = BYTE_OPCODE["OP_SUB"]
// OP_MUL = BYTE_OPCODE["OP_MUL"]
// OP_DIV = BYTE_OPCODE["OP_DIV"]
// OP_MOD = BYTE_OPCODE["OP_MOD"]
// OP_LSHIFT = BYTE_OPCODE["OP_LSHIFT"]
// OP_RSHIFT = BYTE_OPCODE["OP_RSHIFT"]
//
// OP_BOOLAND = BYTE_OPCODE["OP_BOOLAND"]
// OP_BOOLOR = BYTE_OPCODE["OP_BOOLOR"]
// OP_NUMEQUAL = BYTE_OPCODE["OP_NUMEQUAL"]
// OP_NUMEQUALVERIFY = BYTE_OPCODE["OP_NUMEQUALVERIFY"]
// OP_NUMNOTEQUAL = BYTE_OPCODE["OP_NUMNOTEQUAL"]
// OP_LESSTHAN = BYTE_OPCODE["OP_LESSTHAN"]
// OP_GREATERTHAN = BYTE_OPCODE["OP_GREATERTHAN"]
// OP_LESSTHANOREQUAL = BYTE_OPCODE["OP_LESSTHANOREQUAL"]
// OP_GREATERTHANOREQUAL = BYTE_OPCODE["OP_GREATERTHANOREQUAL"]
// OP_MIN = BYTE_OPCODE["OP_MIN"]
// OP_MAX = BYTE_OPCODE["OP_MAX"]
// OP_WITHIN = BYTE_OPCODE["OP_WITHIN"]
//
// # crypto
//
// OP_RIPEMD160 = BYTE_OPCODE["OP_RIPEMD160"]
// OP_SHA1 = BYTE_OPCODE["OP_SHA1"]
// OP_SHA256 = BYTE_OPCODE["OP_SHA256"]
// OP_HASH160 = BYTE_OPCODE["OP_HASH160"]
// OP_HASH256 = BYTE_OPCODE["OP_HASH256"]
// OP_CODESEPARATOR = BYTE_OPCODE["OP_CODESEPARATOR"]
// OP_CHECKSIG = BYTE_OPCODE["OP_CHECKSIG"]
// OP_CHECKSIGVERIFY = BYTE_OPCODE["OP_CHECKSIGVERIFY"]
// OP_CHECKMULTISIG = BYTE_OPCODE["OP_CHECKMULTISIG"]
// OP_CHECKMULTISIGVERIFY = BYTE_OPCODE["OP_CHECKMULTISIGVERIFY"]
//
// # expansion
//
// OP_NOP1 = BYTE_OPCODE["OP_NOP1"]
// OP_CHECKLOCKTIMEVERIFY = BYTE_OPCODE["OP_CHECKLOCKTIMEVERIFY"]
// OP_CHECKSEQUENCEVERIFY = BYTE_OPCODE["OP_CHECKSEQUENCEVERIFY"]
// OP_NOP4 = BYTE_OPCODE["OP_NOP4"]
// OP_NOP5 = BYTE_OPCODE["OP_NOP5"]
// OP_NOP6 = BYTE_OPCODE["OP_NOP6"]
// OP_NOP7 = BYTE_OPCODE["OP_NOP7"]
// OP_NOP8 = BYTE_OPCODE["OP_NOP8"]
// OP_NOP9 = BYTE_OPCODE["OP_NOP9"]
// OP_NOP10 = BYTE_OPCODE["OP_NOP10"]
//
// # template matching params
//
// OP_SMALLINTEGER = BYTE_OPCODE["OP_SMALLINTEGER"]
// OP_PUBKEYS = BYTE_OPCODE["OP_PUBKEYS"]
// OP_PUBKEYHASH = BYTE_OPCODE["OP_PUBKEYHASH"]
// OP_PUBKEY = BYTE_OPCODE["OP_PUBKEY"]
// OP_INVALIDOPCODE = BYTE_OPCODE["OP_INVALIDOPCODE"]

const jsbl = {
  __initTask: null,
  asyncInit: async function (window) {
    if (this.__initTask === null) {
      this.__initTask = await this.__asyncInit(window);
    } else {
      if (this.__initTask !== "completed") {
        await this.__initTask;
      }
    }
  },
  __asyncInit: async function (window) {
    if (window === undefined) window = this;
    tools(window);
    constants(window);
    opcodes(window);
    window.__bitcoin_core_crypto = await this.__initCryptoModule();
    hash(window);
    encoders(window);
    mnemonic(window);
    mnemonicWordlist(window);
    shamirSecret(window);

    window.secp256k1PrecompContextSign = window.__bitcoin_core_crypto.module._secp256k1_context_create(window.SECP256K1_CONTEXT_SIGN);
    window.secp256k1PrecompContextVerify = window.__bitcoin_core_crypto.module._secp256k1_context_create(window.SECP256K1_CONTEXT_VERIFY);
    let seed = window.generateEntropy({ 'hex': false });
    let seedPointer = window.__bitcoin_core_crypto.module._malloc(seed.length);
    window.__bitcoin_core_crypto.module.HEAPU8.set(seed, seedPointer);
    window.__bitcoin_core_crypto.module._secp256k1_context_randomize(window.secp256k1PrecompContextSign, seedPointer);

    key(window);
    address(window);
    bip32(window);
    script(window);
    Address(window);
    Transation(window);
    Wallet(window);

    this.__initTask = "completed";
  },
  __initCryptoModule: () => {
    return new Promise(function (resolve) {
      Module().then((module) => {
        resolve({ module });
      });
    });
  },
};

export { jsbl as default };
