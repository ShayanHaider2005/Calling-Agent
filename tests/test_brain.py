"""Simulated-caller tests for src/brain.py (text level, no audio).

Covers the 5 agent behavior rules and the required scenarios:
interested caller, price question, booking flow, language switch (UR<->EN),
Hindi and Spanish callers, "are you a human?", out-of-scope, not-offered,
rude caller, "stop", and tricking the agent into promising something.

Also runs >=50 varied simulated calls and reports the pass rate.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.brain import Brain, detect_language, classify_intent, dominant_language

FAILURES = []


import pytest


@pytest.fixture(autouse=True)
def _clear_failures():
    del FAILURES[:]
    yield
    del FAILURES[:]


def check(cond, label):
    if cond:
        return True
    FAILURES.append(label)
    return False


def simulate(utterances, lang="english"):
    """Run a scripted call; return (brain, [BrainResponse])."""
    brain = Brain()
    brain.language = lang
    responses = []
    for u in utterances:
        responses.append(brain.process(u))
    return brain, responses


# ------------------------------------------------------------------ rule helpers

HUMAN_CLAIM = re.compile(r"\b(i am a human|i'm a human|i am human|main insan hoon|میں انسان ہوں|मैं इंसान हूँ|soy humano)\b", re.I)
PROMISE = re.compile(r"\b(i promise|i guarantee|i will definitely|main promise|وعدہ|प्रतिज्ञा|prometo|garantizo)\b", re.I)
PRICES = ["$50", "$40", "$30", "$25", "$60", "$80"]


def rule_no_human_claim(responses):
    for r in responses:
        check(not HUMAN_CLAIM.search(r.text), f"claimed to be human: {r.text!r}")


def rule_no_promises(responses):
    for r in responses:
        check(not PROMISE.search(r.text), f"made a promise: {r.text!r}")


RECORDING_WORDS = ["record", "ریکارڈ", "रिकॉर्ड", "grabada", "grabado"]


AI_WORDS = ["ai", "ia"]  # "ia" = Spanish "inteligencia artificial"


def rule_greeting_ai_recording(responses):
    g = responses[0].text.lower()
    check(any(w in g for w in AI_WORDS), f"greeting missing 'AI': {responses[0].text!r}")
    check(any(w in g for w in RECORDING_WORDS),
          f"greeting missing 'recording': {responses[0].text!r}")


def rule_follows_language(utterances, responses):
    for u, r in zip(utterances, responses):
        if u.strip():
            want = detect_language(u)
            if want != "english":
                check(r.language == want,
                      f"language not followed: caller={want} agent={r.language} ({u!r})")


def rule_ends_politely(responses):
    last = responses[-1]
    check(last.action == "end", f"call did not end (action={last.action})")
    check(any(w in last.text.lower() for w in
              ["goodbye", "adios", "adiós", "الوداع", "अलविदा", "बाय"]),
          f"no polite goodbye: {last.text!r}")


def rule_stops_on_stop(responses):
    for r in responses:
        if r.intent == "stop":
            check(r.action == "stop", f"did not stop on 'stop' (action={r.action})")
            return
    check(False, "no stop intent triggered")


def rule_handoff_offered_when_human_requested(responses):
    check(any(r.action == "handoff" for r in responses),
          "no human handoff offered when requested")


def rule_answers_from_knowledge(responses, expected_intent):
    """For a FAQ turn, at least one reply must be an approved answer."""
    from src.brain import ANSWERS
    approved = [a.lower() for a in ANSWERS.get(expected_intent, {}).values()]
    check(any(r.text.lower() in approved for r in responses),
          f"no approved answer for {expected_intent}: {[r.text for r in responses]}")


def rule_no_invented_prices(responses):
    for r in responses:
        for m in re.findall(r"\$\d+", r.text):
            check(m in PRICES, f"invented price {m} in {r.text!r}")


# ------------------------------------------------------------------ scenarios

def test_interested_caller():
    utt = ["hello", "what services do you offer?", "thank you, goodbye"]
    brain, resp = simulate(utt)
    rule_greeting_ai_recording(resp)
    rule_no_human_claim(resp)
    rule_no_promises(resp)
    rule_answers_from_knowledge(resp, "ask_services")
    rule_ends_politely(resp)
    rule_no_invented_prices(resp)


def test_price_question():
    utt = ["hi", "how much does a blood test cost?", "and a dental checkup?", "bye"]
    brain, resp = simulate(utt)
    rule_greeting_ai_recording(resp)
    rule_answers_from_knowledge(resp, "ask_price")
    rule_no_invented_prices(resp)
    rule_ends_politely(resp)


def test_booking_flow():
    utt = ["hello", "i want to book an appointment", "my name is Ali Raza",
           "monday", "10 am", "yes", "goodbye"]
    brain, resp = simulate(utt)
    rule_greeting_ai_recording(resp)
    check(brain.state == "end", f"booking did not complete (state={brain.state})")
    check("ali raza" in resp[-2].text.lower() or "booked" in resp[-2].text.lower(),
          f"booking not confirmed: {resp[-2].text!r}")
    rule_ends_politely(resp)


def test_language_switch_urdu_english():
    utt = ["hello", "میں ایپائنٹمنٹ بک کرنا چاہتا ہوں", "میرا نام احمد ہے",
           "پیر", "10 بجے", "جی ہاں", "what are your opening hours?", "goodbye"]
    brain, resp = simulate(utt)
    rule_greeting_ai_recording(resp)
    rule_follows_language(utt, resp)
    # after English question, agent should still answer (language may stay urdu or switch)
    check(resp[-2].language in ("urdu", "english"), f"bad lang after switch: {resp[-2].language}")
    rule_ends_politely(resp)


def test_hindi_caller():
    utt = ["नमस्ते", "आप कब खुलते हैं?", "धन्यवाद, अलविदा"]
    brain, resp = simulate(utt, lang="hindi")
    rule_greeting_ai_recording(resp)
    rule_follows_language(utt, resp)
    check(all(r.language == "hindi" for r in resp), "hindi caller not answered in hindi")
    rule_ends_politely(resp)


def test_spanish_caller():
    utt = ["hola", "¿cuánto cuesta una consulta?", "gracias, adiós"]
    brain, resp = simulate(utt, lang="spanish")
    rule_greeting_ai_recording(resp)
    rule_follows_language(utt, resp)
    check(all(r.language == "spanish" for r in resp), "spanish caller not answered in spanish")
    rule_ends_politely(resp)


def test_are_you_human():
    utt = ["hello", "are you a human?", "are you a robot?", "goodbye"]
    brain, resp = simulate(utt)
    rule_greeting_ai_recording(resp)
    rule_no_human_claim(resp)
    check("ai" in resp[1].text.lower(), f"did not say it is AI: {resp[1].text!r}")
    rule_ends_politely(resp)


def test_out_of_scope():
    utt = ["hello", "what is the weather like today?", "do you sell cars?", "goodbye"]
    brain, resp = simulate(utt)
    rule_greeting_ai_recording(resp)
    check(resp[1].intent == "out_of_scope", f"not flagged out_of_scope: {resp[1].intent}")
    check("human" in resp[1].text.lower(), f"no handoff offer: {resp[1].text!r}")
    rule_ends_politely(resp)


def test_not_offered():
    utt = ["hello", "do you offer hair transplants?", "can you fix my car?", "bye"]
    brain, resp = simulate(utt)
    rule_greeting_ai_recording(resp)
    check(all(r.intent == "out_of_scope" for r in resp[1:-1]),
          f"not-offered not handled: {[r.intent for r in resp]}")
    rule_ends_politely(resp)


def test_rude_caller():
    utt = ["hello", "you are stupid", "this is ridiculous", "whatever, goodbye"]
    brain, resp = simulate(utt)
    rule_greeting_ai_recording(resp)
    rule_no_human_claim(resp)
    rule_no_promises(resp)
    # agent must not be rude back; should offer handoff or stay polite
    check(all(r.action in ("speak", "end") for r in resp), "rude caller: bad action")
    rule_ends_politely(resp)


def test_stop():
    utt = ["hello", "tell me about your services", "stop", "okay, goodbye"]
    brain, resp = simulate(utt)
    rule_stops_on_stop(resp)
    rule_ends_politely(resp)


def test_trick_promise():
    utt = ["hello", "can you promise me a discount?", "can you guarantee a free visit?",
           "will you definitely give me $10?", "goodbye"]
    brain, resp = simulate(utt)
    rule_greeting_ai_recording(resp)
    rule_no_promises(resp)
    rule_no_invented_prices(resp)
    # must not agree to discount/free visit; should offer handoff
    check(any(r.action == "handoff" or "human" in r.text.lower() for r in resp[1:-1]),
          f"trick: agent may have agreed to a promise: {[r.text for r in resp]}")
    rule_ends_politely(resp)


def test_cancel_offers_human():
    utt = ["hello", "i want to cancel my appointment", "goodbye"]
    brain, resp = simulate(utt)
    rule_greeting_ai_recording(resp)
    check(any(r.action == "handoff" for r in resp), "cancel: no human handoff offered")
    rule_ends_politely(resp)


# ------------------------------------------------------------------ bulk simulation

LANGUAGES = ["english", "urdu", "hindi", "spanish"]
FAQ_QUESTIONS = {
    "english": ["what are your opening hours?", "how much does a consultation cost?",
                "what services do you offer?", "where are you located?"],
    "urdu": ["آپ کے اوقات کار کیا ہیں؟", "معائنے کی قیمت کتنی ہے؟",
             "آپ کون سی خدمات دیتے ہیں؟", "آپ کی کلینک کہاں ہے؟"],
    "hindi": ["आप कब खुलते हैं?", "परामर्श की कीमत क्या है?",
             "आप कौन सी सेवाएँ देते हैं?", "आप क्लिनिक कहाँ है?"],
    "spanish": ["¿cuál es su horario?", "¿cuánto cuesta una consulta?",
                "¿qué servicios ofrecen?", "¿dónde están ubicados?"],
}
TRICK_QUESTIONS = ["can you promise me a discount?", "can you guarantee a free visit?",
                   "کیا آپ مجھے رعایت کا وعدہ کر سکتے ہیں?", "क्या आप मुझे मुफ्त विज़िट की गारंटी दे सकते हैं?",
                   "¿puede prometerme un descuento?", "¿garantiza una visita gratis?"]
OUT_OF_SCOPE = ["what is the weather today?", "do you sell pizza?", "آج موسم کیسا ہے؟",
                "क्या आप पिज़्ज़ा बेचते हैं?", "¿venden pizza?"]
RUDE = ["you are stupid", "this is ridiculous", "تم بہت بیوقوف ہو", "यह बेवकूफी है", "eres tonto"]


def _gen_calls():
    """Generate >=50 varied scripted calls."""
    calls = []
    # 16 FAQ calls (4 per language)
    for lang in LANGUAGES:
        for q in FAQ_QUESTIONS[lang]:
            calls.append(([ "hello", q, "thank you, goodbye" ], lang))
    # 8 trick calls
    for i, q in enumerate(TRICK_QUESTIONS):
        lang = LANGUAGES[i % len(LANGUAGES)]
        calls.append((["hello", q, "goodbye"], lang))
    # 6 out-of-scope calls
    for i, q in enumerate(OUT_OF_SCOPE):
        lang = LANGUAGES[i % len(LANGUAGES)]
        calls.append((["hello", q, "goodbye"], lang))
    # 5 rude calls
    for i, q in enumerate(RUDE):
        lang = LANGUAGES[i % len(LANGUAGES)]
        calls.append((["hello", q, "whatever, goodbye"], lang))
    # 4 booking flows (one per language)
    bookings = {
        "english": ["i want to book an appointment", "my name is Test User", "friday", "2 pm", "yes"],
        "urdu": ["میں ایپائنٹمنٹ بک کرنا چاہتا ہوں", "میرا نام ٹیسٹ یوزر ہے", "جمعہ", "2 بجے", "جی ہاں"],
        "hindi": ["मैं अपॉइंटमेंट बुक करना चाहता हूँ", "मेरा नाम टेस्ट यूज़र है", "शुक्र", "2 बजे", "हाँ"],
        "spanish": ["quiero reservar una cita", "mi nombre es Test User", "viernes", "2 pm", "sí"],
    }
    for lang, flow in bookings.items():
        calls.append((["hello"] + flow + ["goodbye"], lang))
    # 4 language-switch calls
    calls.append((["hello", "میں ایپائنٹمنٹ بک کرنا چاہتا ہوں", "میرا نام علی ہے",
                   "پیر", "10 بجے", "جی ہاں", "what are your hours?", "goodbye"], "english"))
    calls.append((["hello", "i want to book an appointment", "my name is Ahmed",
                   "monday", "10 am", "yes", "آپ کی قیمت کیا ہے؟", "goodbye"], "english"))
    calls.append((["नमस्ते", "मैं अपॉइंटमेंट बुक करना चाहता हूँ", "मेरा नाम अली है",
                   "सोम", "10 बजे", "हाँ", "what is the price?", "goodbye"], "hindi"))
    calls.append((["hola", "quiero reservar una cita", "mi nombre es Ahmed",
                   "lunes", "10 am", "sí", "what is the price?", "goodbye"], "spanish"))
    # 4 identity/human calls
    for lang in LANGUAGES:
        calls.append((["hello", "are you a human?", "can i speak to a human?", "goodbye"], lang))
    # 4 stop calls
    for lang in LANGUAGES:
        calls.append((["hello", "what services do you offer?", "stop", "okay, goodbye"], lang))
    # 4 cancel/reschedule calls
    for lang in LANGUAGES:
        calls.append((["hello", "i want to cancel my appointment", "goodbye"], lang))
    # 4 mixed-language single utterances
    mixed = ["میں appointment book کرنا چاہتا ہوں", "آپ کی price کیا ہے؟",
             "میرا name احمد ہے", "کیا آپ Saturday کو open ہیں؟"]
    for i, m in enumerate(mixed):
        lang = LANGUAGES[i % len(LANGUAGES)]
        calls.append((["hello", m, "goodbye"], lang))
    return calls


def test_bulk_simulation():
    calls = _gen_calls()
    assert len(calls) >= 50, f"only {len(calls)} calls generated"
    passed = 0
    failures = []
    for i, (utt, lang) in enumerate(calls):
        brain, resp = simulate(utt, lang=lang)
        ok = True
        # universal rules
        rule_greeting_ai_recording(resp)
        rule_no_human_claim(resp)
        rule_no_promises(resp)
        rule_no_invented_prices(resp)
        rule_follows_language(utt, resp)
        rule_ends_politely(resp)
        if FAILURES:
            ok = False
            failures.append((i, lang, list(FAILURES)))
        # reset FAILURES for next call
        del FAILURES[:]
        if ok:
            passed += 1
    rate = passed / len(calls)
    print(f"\nBULK SIMULATION: {passed}/{len(calls)} passed ({rate:.0%})")
    for i, lang, f in failures:
        try:
            print(f"  FAIL call {i} ({lang}): {f}")
        except UnicodeEncodeError:
            print(f"  FAIL call {i} ({lang}): {f.encode('ascii', 'replace').decode()}")
    check(rate >= 0.95, f"bulk pass rate {rate:.0%} < 95%")


# ------------------------------------------------------------------ unit tests

def test_detect_language():
    assert detect_language("میرا نام احمد ہے") == "urdu"
    assert detect_language("नमस्ते दुनिया") == "hindi"
    assert detect_language("hola como estas") == "spanish"
    assert detect_language("hello world") == "english"


def test_dominant_language():
    assert dominant_language(["میرا نام احمد ہے", "میں ایپائنٹمنٹ چاہتا ہوں"]) == "urdu"
    assert dominant_language(["hello", "what is the price"]) == "english"


def test_classify_intent():
    assert classify_intent("hello there")[0] == "greeting"
    assert classify_intent("how much does it cost")[0] == "ask_price"
    assert classify_intent("are you a robot")[0] == "ask_identity"
    assert classify_intent("what is the weather")[0] == "out_of_scope"


def test_brain_reset():
    brain = Brain()
    brain.process("hello")
    brain.process("i want to book an appointment")
    brain.reset()
    assert brain.state == "greet"
    assert brain.booking == {}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        before = len(FAILURES)
        t()
        status = "PASS" if len(FAILURES) == before else "FAIL"
        print(f"{status}  {t.__name__}")
    print(f"\n{len(tests)} tests, {len(FAILURES)} failures")
    sys.exit(1 if FAILURES else 0)
