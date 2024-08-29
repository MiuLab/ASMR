
REQUEST_2 = """
Here is a reflected action from B.
B: {reflected_action}
A is another person talking to B in a room
What ambiguous request may A talk to B indirectly without asking a question causing B to respond above reflected action.
And describe some related object in the background according to utterance between A and B
The output should be {num} examples in a JSON list
Each example is an object in the list
example output
[
    {{
        "Person A" : "<ambiguous_statement>",
        "Person B" : "{reflected_action}",
        "BackgroundObject" : "<background_object_description>"
    }}
]
## Output
"""

LOCATION_1 = """
Give me {num} conversation examples between two people in a {location}
Person A made an ambiguous request to Person B
And Person B responded with a reflected action to A
Each conversation should be one utterance
And describe some related object in the background
The output should be only in a JSON list
Each example is an object in the list
Example output
[
    {{
        "Person A" : "<ambiguous_request>"
        "Person B" : "<reflected_action>"
        "BackgroundObject" : "<background_object_description>"
    }},
]
## Output
"""

LOCATION_2 = """
Give me {num} conversation examples between two people in a {location}
Person A made an ambiguous statement without asking a question to Person B
And Person B responded with a reflected action to A
Each conversation should be one utterance
And describe some related object in the background
The output should be only in a JSON list
Each example is an object in the list
Example output
[
    {{
        "Person A" : "<ambiguous_statement>"
        "Person B" : "<reflected_action>"
        "BackgroundObject" : "<background_object_description>"
    }},
]
## Output
"""

LOCATION_3 = """
Give me {num} conversation examples between two people in a {location}
Person A made an ambiguous request indirectly without asking a question to Person B
And Person B responded with a reflected action to A
Each conversation should be one utterance
And describe some related object in the background
The output should be only in a JSON list
Each example is an object in the list
Example output
[
    {{
        "Person A" : "<ambiguous_statement>"
        "Person B" : "<reflected_action>"
        "BackgroundObject" : "<background_object_description>"
    }},
]
## Output
"""
