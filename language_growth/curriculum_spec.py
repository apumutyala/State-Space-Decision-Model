"""Predeclared synthetic curriculum and fresh transfer wording; no PR #67 data."""

CURRICULUM_SPLITS = {
    "development_curriculum": {
        "names": ("Aster", "Bracken", "Cinder", "Dovetail", "Ember", "Fallow"),
        "warranty_range": (1, 36), "price_cents_range": (5000, 20000),
        "templates": {
            "read": (
                "What is {entity}'s {field} in October 2026?",
                "Give the October 2026 {field} for {entity}.",
                "Read the current {field} recorded for {entity}.",
                "Report the value of {field} belonging to {entity} this October.",
                "Look up {entity} and return its {field}.",
                "Tell me the recorded {field} of {entity} for this month.",
                "I need {entity}'s October 2026 {field} value.",
                "Retrieve {field} from {entity}'s October record.",
            ),
            "explain": (
                "Explain the checked {field} fact for {entity}.",
                "Show the checked derivation for {entity}'s {field}.",
                "Explain how the record supports {entity}'s {field}.",
                "Give the evidence steps establishing the {field} of {entity}.",
                "Justify the asserted {field} for {entity} with a derivation.",
                "Walk through the proof of {entity}'s recorded {field}.",
                "Show why the {field} statement about {entity} is supported.",
                "Trace the checked reasoning behind {entity}'s {field} fact.",
            ),
            "compare": (
                "Compare the October {field} of {left} and {right}.",
                "How do {left}'s and {right}'s October {field} compare?",
                "Determine the {field} relation between {left} and {right}.",
                "Contrast {left} with {right} using their recorded {field}.",
                "Is {left}'s {field} greater, smaller, or equal to {right}'s?",
                "Check the ordering of {field} for {left} versus {right}.",
                "Report how {left} stands relative to {right} on {field}.",
                "Use the October records to compare {left} against {right} on {field}.",
            ),
        },
    },
    "development_transfer_v2": {
        "names": ("Marigold", "Nettle", "Orchid", "Plover", "Quince", "Rowan"),
        "warranty_range": (97, 132), "price_cents_range": (45001, 60000),
        "templates": {
            "read": (
                "From the October ledger, supply {entity}'s {field}.",
                "Which {field} value does the October entry assign to {entity}?",
                "Please quote the stored {field} associated with {entity}.",
                "For {entity}, fetch the {field} datum dated October 2026.",
                "Find {entity}'s entry and state its {field} amount.",
                "The requested datum is {field}; the subject is {entity}.",
            ),
            "explain": (
                "Account for the support linking {entity} to its {field} fact.",
                "What proof connects {entity}'s ledger entry to the {field} assertion?",
                "For the {field} claim concerning {entity}, lay out its justification.",
                "Describe the derivation that licenses {entity}'s {field} statement.",
                "Unpack the evidence chain for {entity} having the recorded {field}.",
                "Demonstrate the support for the October {field} assertion about {entity}.",
            ),
            "compare": (
                "Establish whether {left} precedes, ties, or exceeds {right} in {field}.",
                "Taking {left} first and {right} second, state their {field} ordering.",
                "What ordering holds for the {field} entries of {left} versus {right}?",
                "Set {left}'s October {field} beside {right}'s and identify the relation.",
                "Resolve the relative magnitude of {field}: {left} followed by {right}.",
                "For the pair {left}, {right}, establish the October {field} relationship.",
            ),
        },
    },
}
