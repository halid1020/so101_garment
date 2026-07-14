# User-study session pack — participant P4

Experimenter copy. Protocol: `documents/user_study_protocol.md` (design in
the teleoperation paper). Fill every blank during the session; file the
pack afterwards. Condition order for P4 (balanced Latin square):
**C4 -> C1 -> C3 -> C2**, presented to the participant as A -> B -> C -> D.

| Field | Value |
|---|---|
| Participant ID | P4 |
| Date / time |  |
| Experimenter |  |
| Note-taker (optional) |  |

## 1. Consent and demographics (5 min)

- [ ] Purpose explained; right to stop at any time without reason
- [ ] What is recorded explained (timings, console logs, questionnaires, interview notes)
- [ ] Consent given
- [ ] Safety brief: kill switch location; stand clear of the arms' reach

| Demographic | Value |
|---|---|
| Age range |  |
| Handedness |  |
| Prior VR experience (1–5) |  |
| Prior teleoperation experience (1–5) |  |
| Prior gaming experience (1–5) |  |

## 2. Familiarisation (10 min, simulation)

`venv/bin/python tool/quest_sim_teleop.py --method pink_relaxed`

- [ ] Clutch explained (hold both grips; point handles down at each grip)
- [ ] Triggers close the grippers
- [ ] Joystick clicks reset that gripper's roll at the next grip
- [ ] Headset-anywhere explained (placement does not matter)
- [ ] Participant can move both end-effectors deliberately

## Condition A (experimenter key: C4 — native upstream Telegrip stack (own UI + IK, no envelope))

Refer to this condition only as **"A"** in front of the participant.

Launch: `venv/bin/python tool/telegrip_native.py --autoconnect`
Console log (C1–C3): append `2>&1 | tee outputs/user_study/<PID>_C4.log`

- [ ] Condition launched and arms respond
- [ ] ~3 min free practice with the cube (not scored)

### Trials (primary: pick–handover–place; timeout 3 min each)

| Trial | Time (s) | Success (Y/N) | Failure mode | Drops | Re-grips | OOE time (s) | Rig collisions |
|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |
| Towel fold |  |  |  |  |  |  |  |

### NASA-TLX (0–100 each; completed immediately after the trials)

| # | Item | Rating (0-100) |
|---|---|---|
| 1 | Mental demand |  |
| 2 | Physical demand |  |
| 3 | Temporal demand |  |
| 4 | Performance |  |
| 5 | Effort |  |
| 6 | Frustration |  |

### SUS (1 = strongly disagree … 5 = strongly agree)

| # | Item | Rating (1-5) |
|---|---|---|
| 1 | I think that I would like to use this system frequently. |  |
| 2 | I found the system unnecessarily complex. |  |
| 3 | I thought the system was easy to use. |  |
| 4 | I think that I would need the support of a technical person to be able to use this system. |  |
| 5 | I found the various functions in this system were well integrated. |  |
| 6 | I thought there was too much inconsistency in this system. |  |
| 7 | I would imagine that most people would learn to use this system very quickly. |  |
| 8 | I found the system very cumbersome to use. |  |
| 9 | I felt very confident using the system. |  |
| 10 | I needed to learn a lot of things before I could get going with this system. |  |

### Teleoperation-feel items (1 = strongly disagree … 7 = strongly agree)

| # | Item | Rating (1-7) |
|---|---|---|
| 1 | The grippers went where I intended (precision). |  |
| 2 | The wrist followed my hand rotation without noticeable delay (wrist agility). |  |
| 3 | The system behaved predictably when I reached the edge of the arms' range (boundary behaviour). |  |
| 4 | Releasing and re-gripping to re-centre my hands was easy (clutch). |  |
| 5 | I could position myself/the headset where I wanted and still control comfortably (headset-anywhere). |  |
| 6 | Coordinating both arms at the same time felt manageable (bimanual). |  |

- [ ] Short break offered; arms to rest pose

## Condition B (experimenter key: C1 — armplane pipeline, `pink_full` tracker, OOE `project`)

Refer to this condition only as **"B"** in front of the participant.

Launch: `venv/bin/python tool/meta_quest_teleopration.py --method pink_full --oob-mode project`
Console log (C1–C3): append `2>&1 | tee outputs/user_study/<PID>_C1.log`

- [ ] Condition launched and arms respond
- [ ] ~3 min free practice with the cube (not scored)

### Trials (primary: pick–handover–place; timeout 3 min each)

| Trial | Time (s) | Success (Y/N) | Failure mode | Drops | Re-grips | OOE time (s) | Rig collisions |
|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |
| Towel fold |  |  |  |  |  |  |  |

### NASA-TLX (0–100 each; completed immediately after the trials)

| # | Item | Rating (0-100) |
|---|---|---|
| 1 | Mental demand |  |
| 2 | Physical demand |  |
| 3 | Temporal demand |  |
| 4 | Performance |  |
| 5 | Effort |  |
| 6 | Frustration |  |

### SUS (1 = strongly disagree … 5 = strongly agree)

| # | Item | Rating (1-5) |
|---|---|---|
| 1 | I think that I would like to use this system frequently. |  |
| 2 | I found the system unnecessarily complex. |  |
| 3 | I thought the system was easy to use. |  |
| 4 | I think that I would need the support of a technical person to be able to use this system. |  |
| 5 | I found the various functions in this system were well integrated. |  |
| 6 | I thought there was too much inconsistency in this system. |  |
| 7 | I would imagine that most people would learn to use this system very quickly. |  |
| 8 | I found the system very cumbersome to use. |  |
| 9 | I felt very confident using the system. |  |
| 10 | I needed to learn a lot of things before I could get going with this system. |  |

### Teleoperation-feel items (1 = strongly disagree … 7 = strongly agree)

| # | Item | Rating (1-7) |
|---|---|---|
| 1 | The grippers went where I intended (precision). |  |
| 2 | The wrist followed my hand rotation without noticeable delay (wrist agility). |  |
| 3 | The system behaved predictably when I reached the edge of the arms' range (boundary behaviour). |  |
| 4 | Releasing and re-gripping to re-centre my hands was easy (clutch). |  |
| 5 | I could position myself/the headset where I wanted and still control comfortably (headset-anywhere). |  |
| 6 | Coordinating both arms at the same time felt manageable (bimanual). |  |

- [ ] Short break offered; arms to rest pose

## Condition C (experimenter key: C3 — armplane pipeline, `telegrip` split IK, OOE `project`)

Refer to this condition only as **"C"** in front of the participant.

Launch: `venv/bin/python tool/meta_quest_teleopration.py --method telegrip --oob-mode project`
Console log (C1–C3): append `2>&1 | tee outputs/user_study/<PID>_C3.log`

- [ ] Condition launched and arms respond
- [ ] ~3 min free practice with the cube (not scored)

### Trials (primary: pick–handover–place; timeout 3 min each)

| Trial | Time (s) | Success (Y/N) | Failure mode | Drops | Re-grips | OOE time (s) | Rig collisions |
|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |
| Towel fold |  |  |  |  |  |  |  |

### NASA-TLX (0–100 each; completed immediately after the trials)

| # | Item | Rating (0-100) |
|---|---|---|
| 1 | Mental demand |  |
| 2 | Physical demand |  |
| 3 | Temporal demand |  |
| 4 | Performance |  |
| 5 | Effort |  |
| 6 | Frustration |  |

### SUS (1 = strongly disagree … 5 = strongly agree)

| # | Item | Rating (1-5) |
|---|---|---|
| 1 | I think that I would like to use this system frequently. |  |
| 2 | I found the system unnecessarily complex. |  |
| 3 | I thought the system was easy to use. |  |
| 4 | I think that I would need the support of a technical person to be able to use this system. |  |
| 5 | I found the various functions in this system were well integrated. |  |
| 6 | I thought there was too much inconsistency in this system. |  |
| 7 | I would imagine that most people would learn to use this system very quickly. |  |
| 8 | I found the system very cumbersome to use. |  |
| 9 | I felt very confident using the system. |  |
| 10 | I needed to learn a lot of things before I could get going with this system. |  |

### Teleoperation-feel items (1 = strongly disagree … 7 = strongly agree)

| # | Item | Rating (1-7) |
|---|---|---|
| 1 | The grippers went where I intended (precision). |  |
| 2 | The wrist followed my hand rotation without noticeable delay (wrist agility). |  |
| 3 | The system behaved predictably when I reached the edge of the arms' range (boundary behaviour). |  |
| 4 | Releasing and re-gripping to re-centre my hands was easy (clutch). |  |
| 5 | I could position myself/the headset where I wanted and still control comfortably (headset-anywhere). |  |
| 6 | Coordinating both arms at the same time felt manageable (bimanual). |  |

- [ ] Short break offered; arms to rest pose

## Condition D (experimenter key: C2 — armplane pipeline, `pink_relaxed` tracker, OOE `project`)

Refer to this condition only as **"D"** in front of the participant.

Launch: `venv/bin/python tool/meta_quest_teleopration.py --method pink_relaxed --oob-mode project`
Console log (C1–C3): append `2>&1 | tee outputs/user_study/<PID>_C2.log`

- [ ] Condition launched and arms respond
- [ ] ~3 min free practice with the cube (not scored)

### Trials (primary: pick–handover–place; timeout 3 min each)

| Trial | Time (s) | Success (Y/N) | Failure mode | Drops | Re-grips | OOE time (s) | Rig collisions |
|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |
| Towel fold |  |  |  |  |  |  |  |

### NASA-TLX (0–100 each; completed immediately after the trials)

| # | Item | Rating (0-100) |
|---|---|---|
| 1 | Mental demand |  |
| 2 | Physical demand |  |
| 3 | Temporal demand |  |
| 4 | Performance |  |
| 5 | Effort |  |
| 6 | Frustration |  |

### SUS (1 = strongly disagree … 5 = strongly agree)

| # | Item | Rating (1-5) |
|---|---|---|
| 1 | I think that I would like to use this system frequently. |  |
| 2 | I found the system unnecessarily complex. |  |
| 3 | I thought the system was easy to use. |  |
| 4 | I think that I would need the support of a technical person to be able to use this system. |  |
| 5 | I found the various functions in this system were well integrated. |  |
| 6 | I thought there was too much inconsistency in this system. |  |
| 7 | I would imagine that most people would learn to use this system very quickly. |  |
| 8 | I found the system very cumbersome to use. |  |
| 9 | I felt very confident using the system. |  |
| 10 | I needed to learn a lot of things before I could get going with this system. |  |

### Teleoperation-feel items (1 = strongly disagree … 7 = strongly agree)

| # | Item | Rating (1-7) |
|---|---|---|
| 1 | The grippers went where I intended (precision). |  |
| 2 | The wrist followed my hand rotation without noticeable delay (wrist agility). |  |
| 3 | The system behaved predictably when I reached the edge of the arms' range (boundary behaviour). |  |
| 4 | Releasing and re-gripping to re-centre my hands was easy (clutch). |  |
| 5 | I could position myself/the headset where I wanted and still control comfortably (headset-anywhere). |  |
| 6 | Coordinating both arms at the same time felt manageable (bimanual). |  |

- [ ] Short break offered; arms to rest pose

## 3. Closing (10 min)

Forced ranking, best -> worst overall (participant's labels):

| Rank 1 | Rank 2 | Rank 3 | Rank 4 |
|---|---|---|---|
|  |  |  |  |

### Semi-structured interview (recorded or noted)

**Q.** Walk me through how the whole process felt, from putting on the headset to finishing a trial.

>

**Q.** What felt most natural? Least natural?

>

**Q.** Was there a moment you lost confidence in what the robot would do? What happened?

>

**Q.** Did you notice differences between A-D? Which, and how did you adapt?

>

**Q.** How did the two-arm coordination feel compared to what you expected?

>

**Q.** If you could change one thing about the controls, what?

>

## 4. Debrief

- [ ] Thanked; conditions revealed if asked
- [ ] Nausea/fatigue check passed throughout (session stopped on any report)
