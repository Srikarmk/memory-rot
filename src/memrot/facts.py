"""Value pools and phrasing templates for the synthetic user histories.

Every value of a probed attribute is a distinctive string, so the judge can tell
which value an answer gives by matching strings. Templates use {v} for the
value, {old} for the previous value and {p} for the project name.
"""

ATTRIBUTES = ["city", "manager", "deadline", "notes_tool", "diet", "meeting_day"]

CITIES = ["Denver", "Boston", "Austin", "Seattle", "Chicago", "Atlanta", "Minneapolis", "Pittsburgh",
          "Raleigh", "Phoenix", "San Diego", "Nashville", "Baltimore", "Sacramento", "Milwaukee", "Tucson",
          "Cleveland", "Omaha"]

FIRST_NAMES = ["Priya", "Dana", "Marcus", "Ingrid", "Tomas", "Yuki", "Leandra", "Oskar", "Femi", "Hollis",
               "Renata", "Bastian", "Imogen", "Caspian", "Odalys", "Thaddeus", "Wren", "Anselm", "Mirela",
               "Deshawn", "Solveig", "Ezra", "Lucinda", "Kofi"]
LAST_NAMES = ["Okonkwo", "Whitfield", "Lindqvist", "Arceneaux", "Brandauer", "Castellanos", "Dimitrova",
              "Fairbanks", "Gallowglass", "Halvorsen", "Ibarguen", "Jaskolski", "Kettleborough", "Larkspur",
              "Montgomery", "Nakashima", "Oyelaran", "Pemberton", "Quintanilla", "Rasmussen"]

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
MONTH_ABBR = {m: m[:3] for m in MONTHS}

NOTES_TOOLS = ["Obsidian", "Notion", "Evernote", "OneNote", "Logseq", "Joplin", "Zettlr", "Trilium",
               "Simplenote", "Notability"]

DIETS = ["vegetarian", "pescatarian", "gluten-free", "dairy-free", "keto", "halal", "kosher"]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

PROJECTS = ["Halcyon", "Bluefin", "Marigold", "Cobalt", "Juniper", "Meridian", "Tidewater", "Quillon",
            "Saffron", "Barnacle", "Kestrel", "Ironwood", "Pinwheel", "Driftwood"]

USER_NAMES = ["Avery", "Jordan", "Riley", "Morgan", "Casey", "Quinn", "Rowan", "Sasha", "Emerson",
              "Harper", "Reese", "Skyler", "Finley", "Tatum", "Blair", "Hayden"]

# Ordinary statements. Implied updates reuse these: the new value is simply
# mentioned the way an established fact would be, with nothing marking a change.
STATEMENT = {
    "city": [
        "I live in {v}.",
        "For context, I'm based in {v}.",
        "It's been raining all week here in {v}.",
        "My apartment in {v} is pretty small, so keep that in mind.",
        "Since I live in {v}, please give times in my local time zone.",
        "I'm looking for a good running route near my place in {v}.",
        "Home for me is {v}.",
        "The farmers market in my neighborhood in {v} opens again this weekend.",
    ],
    "manager": [
        "My manager, {v}, wants the status report by the end of the week.",
        "I have my weekly one-on-one with my manager {v} tomorrow.",
        "I report to {v}.",
        "{v}, my manager, asked me to lead the next sprint review.",
        "My boss is {v}, who likes short written updates.",
        "I need to send my goals for the quarter to {v}, my manager.",
        "My manager {v} approved my vacation request.",
    ],
    "deadline": [
        "The deadline for Project {p} is {v}.",
        "Project {p} is due {v}, so I'm pretty busy.",
        "We need to ship Project {p} by {v}.",
        "Everything for Project {p} has to be finished by {v}.",
        "I'm counting down to {v}, the Project {p} deadline.",
        "The Project {p} launch date is {v}.",
    ],
    "notes_tool": [
        "I keep all my notes in {v}.",
        "I'll paste this into {v}, which is where I keep my notes.",
        "{v} is my note-taking app of choice.",
        "My notes live in {v}, so plain text with headings works best.",
        "I write my meeting notes in {v}.",
        "Everything I jot down goes into {v}.",
    ],
    "diet": [
        "I'm {v}, so please keep food suggestions {v}.",
        "Please only suggest {v} recipes for me.",
        "I follow a {v} diet.",
        "When you suggest restaurants, remember that I eat {v}.",
        "My diet is {v}.",
        "I stick to {v} food, so plan meals around that.",
    ],
    "meeting_day": [
        "Our weekly team meeting is on {v}.",
        "I need slides ready for our {v} team meeting.",
        "The team meets every {v}.",
        "My weekly team meeting happens on {v} mornings.",
        "I block out {v} morning for our team meeting.",
        "We have our team meeting each {v}.",
    ],
}

# Explicit updates say that something changed.
EXPLICIT_UPDATE = {
    "city": [
        "Big news: I moved to {v} last weekend.",
        "I've relocated, and I live in {v} now.",
        "I no longer live in {old}; I moved to {v}.",
        "Update for you: I moved from {old} to {v}.",
        "I just finished moving, so home is {v} now.",
    ],
    "manager": [
        "Reorg news: my new manager is {v}.",
        "I no longer report to {old}; my manager is now {v}.",
        "{old} moved to another team, so I report to {v} now.",
        "I have a new boss: {v}.",
        "My manager changed this week, and it's {v} now.",
    ],
    "deadline": [
        "The Project {p} deadline got moved to {v}.",
        "Change of plans: Project {p} is now due {v} instead of {old}.",
        "Our client pushed the Project {p} deadline to {v}.",
        "Project {p} has a new deadline: {v}.",
        "Heads up, the Project {p} due date changed to {v}.",
    ],
    "notes_tool": [
        "I switched my notes over to {v}.",
        "I stopped using {old} and moved all my notes to {v}.",
        "New setup: I keep my notes in {v} now.",
        "I've changed note apps, and I use {v} now.",
        "I migrated everything from {old} to {v} for my notes.",
    ],
    "diet": [
        "I'm not {old} anymore; I'm {v} now.",
        "I stopped following a {old} diet, and I'm {v} now.",
        "I dropped the {old} diet and switched to {v}.",
        "My doctor suggested a change, so I eat {v} now instead of {old}.",
        "I switched from {old} to {v} food this month.",
    ],
    "meeting_day": [
        "Our weekly team meeting moved to {v}.",
        "Heads up, the team meeting is on {v} now instead of {old}.",
        "The team meeting changed days, and it's every {v} now.",
        "We rescheduled the weekly team meeting to {v}.",
        "Starting this week the team meets on {v}.",
    ],
}

QUESTION = {
    "city": ["Which city do I live in?", "Where am I based?", "What city is home for me?"],
    "manager": ["Who is my manager?", "Who do I report to?", "What's my manager's name?"],
    "deadline": ["When is the Project {p} deadline?", "When is Project {p} due?",
                 "What's the due date for Project {p}?"],
    "notes_tool": ["Which app do I use for my notes?", "Where do I keep my notes?",
                   "What note-taking app do I use?"],
    "diet": ["What diet do I follow?", "What kind of food do I eat?", "What are my dietary restrictions?"],
    "meeting_day": ["What day is my weekly team meeting?", "Which day does my team meet?",
                    "On what day is our weekly team meeting?"],
}

# Personal facts that sit close to a probed attribute but are about something
# else. {x} is drawn from the same pool as the attribute, never one of its values.
NEAR_MISS = {
    "city": ["My sister lives in {x}.", "I'm flying to {x} next month for a conference.",
             "My best friend just bought a house in {x}.", "My parents retired to {x}."],
    "manager": ["My coworker {x} is great with spreadsheets.", "I'm mentoring a new hire named {x}.",
                "{x} from the design team sent me a mockup.", "My neighbor {x} lent me a ladder."],
    "deadline": ["My friend's wedding is on {x}.", "My passport expires on {x}.",
                 "The lease on my car ends {x}.", "My cousin's graduation is {x}."],
    "notes_tool": ["My teammate swears by {x} for notes, but it's not for me.",
                   "I tried {x} once and didn't like it.", "My sister keeps her recipes in {x}."],
    "diet": ["My roommate is {x}, so I sometimes cook for two.", "My dad recently went {x}.",
             "I'm hosting a friend who eats {x} this weekend."],
    "meeting_day": ["I go to the climbing gym every {x}.", "My piano lesson is on {x} evenings.",
                    "I call my grandmother every {x}."],
}

# Personal facts unrelated to any probed attribute.
PERSONAL = [
    "I've been learning to play the ukulele.", "I adopted a cat named Miso last year.",
    "I'm training for a half marathon.", "My favorite author is Ursula K. Le Guin.",
    "I've started doing pottery on weekends.", "I'm trying to drink less coffee.",
    "I just finished a jigsaw puzzle with two thousand pieces.", "I'm saving up for a new bike.",
    "My niece turns six soon.", "I've been getting into birdwatching.",
    "I'm reading a lot of science fiction lately.", "My laptop battery barely lasts two hours now.",
    "I want to learn some basic Portuguese this year.", "My balcony garden finally has tomatoes.",
    "I'm terrible at remembering birthdays.", "I've been sleeping badly this week.",
]

# Requests with no personal information in them.
TASKS = [
    "Can you explain the difference between a list and a tuple in Python?",
    "What's a good way to structure a one-page project proposal?",
    "Give me three tips for writing clearer emails.",
    "How do I calculate compound interest?",
    "What are some good stretches after sitting all day?",
    "Can you summarize the main idea of opportunity cost?",
    "How long should I steep green tea?",
    "What's the difference between a virus and a bacterium?",
    "Suggest a name for a bookshelf-building tutorial.",
    "How do I convert Celsius to Fahrenheit in my head?",
    "What's a polite way to decline a meeting?",
    "Explain what a hash table is in two sentences.",
    "What are some good questions to ask at the end of an interview?",
    "How do I get candle wax out of a tablecloth?",
    "What's the rule for when to use 'fewer' versus 'less'?",
    "Can you recommend a simple weekly workout plan?",
    "Why does bread go stale?",
    "Give me a quick checklist for a code review.",
    "What's the best way to memorize a short speech?",
    "How do noise-cancelling headphones work?",
    "Write a two-line thank-you note for a colleague.",
    "What are the pros and cons of standing desks?",
    "How do I keep basil alive indoors?",
    "Explain the difference between median and mean.",
]

# Short requests that wrap fact statements, to make sessions read like real chats.
WRAPPERS = [
    "Can you help me plan my week?", "Any tips for staying focused this afternoon?",
    "I could use some help organizing my to-do list.", "Can you draft a short message to my team?",
    "What should I prioritize today?", "Help me think through my schedule.",
    "Could you suggest a dinner idea?", "I need a quick pep talk.", "",
    "", "",
]

OPENERS = ["", "", "Hi again.", "Morning!", "Hey.", "Quick one.", "Hello!"]
