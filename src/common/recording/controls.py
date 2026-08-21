"""How a collection session is driven, in one list.

The terminal prints these steps when teleoperation starts
(``tool/meta_quest_teleopration.py``) and the rig console shows them beside its
live view, so a rig with no display in front of it is driven from the same
instructions as one with a keyboard attached.

``where`` is the point of the list. Almost every step moves an arm, and those
stay on the headset or on the keyboard beside the rig, where the operator can
see what is about to move; only the two the browser may take say so, and they
are exactly the two keys ``monitor_server.DEFAULT_ALLOWED_KEYS`` admits.

Pure: no hardware, no aiohttp, no imports beyond the standard library.
"""

from __future__ import annotations

HEADSET = "headset"
KEYBOARD = "session keyboard"
CONTROLLERS = "controllers"
LEADERS = "leader arms"


def control_steps(
    input_mode: str = "quest", record: bool = True, method: str = ""
) -> "list[dict[str, str]]":
    """How this session is driven, in order. Pure — unit-tested.

    Each step is ``{key, what, where}``; ``key`` is empty for the steps that are
    not a key press at all (holding the grips, moving the leader arms). The two
    steps the browser may take say so in ``where``, and they are exactly the two
    keys ``DEFAULT_ALLOWED_KEYS`` admits.
    """
    leader = input_mode == "leader"
    surface = KEYBOARD if leader else HEADSET
    episode = (
        "start recording an episode; press again to save it"
        if record
        else "records an episode — this session was started without recording"
    )
    steps: list[dict[str, str]] = [
        {
            "key": "Y",
            "what": "enable both arms: they move to the ready pose and hold it",
            "where": surface,
        }
    ]
    if leader:
        steps += [
            {
                "key": "",
                "what": "move the leader arms — the followers mirror them",
                "where": LEADERS,
            },
            {
                "key": "",
                "what": "squeeze a leader jaw to close that follower's gripper",
                "where": LEADERS,
            },
        ]
    else:
        steps += [
            {
                "key": "",
                "what": "hold BOTH grips to teleoperate — the arms follow your hands",
                "where": CONTROLLERS,
            },
            {
                "key": "",
                "what": "hold a trigger to close that gripper",
                "where": CONTROLLERS,
            },
        ]
        if method == "mymethod":
            steps.append(
                {
                    "key": "",
                    "what": "deflect a thumbstick to trim that wrist "
                    "(x = roll, y = flex); the handle resumes on release",
                    "where": CONTROLLERS,
                }
            )
    steps += [
        {"key": "A", "what": episode, "where": f"{surface} or this page"},
        {"key": "B", "what": "send both arms back to the ready pose", "where": surface},
        {"key": "X", "what": "park: rest pose, torque off", "where": surface},
        {
            "key": "Q",
            "what": "end the session — it parks the arms, finishes any episode "
            "in progress and closes the dataset",
            "where": f"{surface} or this page",
        },
    ]
    return steps
