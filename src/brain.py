"""Brain: state machine + intent classification + guardrails for the clinic scenario.

Works at the TEXT level (no audio) so it is fully testable.
The language model is OPTIONAL: it only (a) classifies intent and (b) lightly
rephrases approved answers. A rule-based fallback works with no LLM at all.

Pipeline position: STT text -> Brain -> reply text (+ language + action).
"""
import os
import re

import yaml

# ------------------------------------------------------------------ languages

URDU_SCRIPT = re.compile(r"[\u0600-\u06FF]")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
SPANISH_WORDS = {"el", "la", "los", "las", "un", "una", "por", "favor", "gracias",
                 "cita", "horario", "precio", "dónde", "donde", "qué", "que",
                 "cuánto", "cuanto", "está", "esta", "abierto", "reservar",
                 "doctor", "médico", "medico", "teléfono", "telefono", "hola",
                 "buenos", "días", "dias", "tarde", "necesito", "quiero", "puedo"}
ENGLISH_WORDS = {"the", "is", "are", "please", "thank", "book", "appointment",
                 "what", "where", "how", "when", "can", "could", "would", "i",
                 "you", "your", "open", "close", "hours", "price", "cost",
                 "doctor", "clinic", "hello", "hi", "good", "morning",
                 "afternoon", "need", "want", "speak", "human", "robot"}


def detect_language(text):
    """Return 'urdu', 'hindi', 'spanish', or 'english'.

    Script is decisive for Urdu/Hindi. For Latin script, count stopwords;
    ties go to English (the demo's default).
    """
    if not text:
        return "english"
    if URDU_SCRIPT.search(text):
        return "urdu"
    if DEVANAGARI.search(text):
        return "hindi"
    words = set(re.findall(r"[a-záéíóúñü]+", text.lower()))
    es = len(words & SPANISH_WORDS)
    en = len(words & ENGLISH_WORDS)
    if es > en:
        return "spanish"
    return "english"


def dominant_language(texts):
    """Dominant language across a list of utterances (for mixed speech)."""
    counts = {}
    for t in texts:
        lang = detect_language(t)
        counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return "english"
    return max(counts, key=counts.get)


# ------------------------------------------------------------------ intents

INTENT_KEYWORDS = {
    "greeting": ["hello", "hi", "salam", "salaam", "adaab", "namaste", "hola",
                 "buenos", "good morning", "good afternoon"],
    "ask_hours": ["hour", "open", "close", "time", "when", "timing", "اوقات",
                  "کھلا", "کب", "horario", "abierto", "cerrado", "खुला", "समय"],
    "ask_price": ["price", "cost", "fee", "charge", "how much", "قیمت", "کتنی",
                  "kitna", "kitni", "precio", "cuánto", "costo", "कीमत"],
    "ask_services": ["service", "offer", "treatment", "treat", "خدمات", "کون سی",
                     "servicio", "सेवा"],
    "ask_location": ["where", "location", "address", "located", "کہاں", "پتہ",
                     "donde", "dirección", "कहां", "पता"],
    "book_appointment": ["book", "appointment", "schedule", "visit", "ایپائنٹمنٹ",
                        "بک", "appointment", "cita", "reservar", "अपॉइंटमेंट", "बुक"],
    "cancel_appointment": ["cancel", "منسوخ", "cancelar", "रद्द"],
    "reschedule_appointment": ["reschedule", "change", "move", "تبدیل", "cambiar",
                               "बदल"],
    "ask_human": ["human", "person", "real", "agent", "someone", "انسان", "شخص",
                  "persona", "इंसान", "मनुष्य"],
    "ask_identity": ["robot", "ai", "artificial", "machine", "computer", "روبوٹ",
                     "مشین", "robot", "máquina", "रोबोट", "मशीन"],
    "goodbye": ["goodbye", "bye", "alwida", "الوداع", "adios", "अलविदा", "बाय"],
    "stop": ["stop", "stop talking", "ruk", "رکو", "stop it"],
}


def classify_intent(text):
    """Rule-based intent classification. Returns (intent, matched_keyword)."""
    t = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return intent, kw
    return "out_of_scope", None


# ------------------------------------------------------------------ answers

ANSWERS = {
    "greeting": {
        "english": "Hello, thank you for calling Sunrise Family Clinic. This is an AI assistant, and this call may be recorded. How can I help you today?",
        "urdu": "السلام علیکم، سنرے فیملی کلینک پر کال کرنے کا شکریہ۔ یہ ایک AI اسسٹنٹ ہے اور یہ کال ریکارڈ ہو سکتی ہے۔ آج میں آپ کی کیسے مدد کر سکتا ہوں؟",
        "hindi": "नमस्ते, सनराइज़ फैमिली क्लिनिक पर कॉल करने के लिए धन्यवाद। यह एक AI सहायक है और यह कॉल रिकॉर्ड हो सकती है। आज मैं आपकी कैसे मदद कर सकता हूँ?",
        "spanish": "Hola, gracias por llamar a la Clínica Familiar Sunrise. Este es un asistente de IA y esta llamada puede ser grabada. ¿Cómo puedo ayudarle hoy?",
    },
    "ask_hours": {
        "english": "We are open Monday to Friday from 9 AM to 5 PM, and on Saturday from 10 AM to 2 PM. We are closed on Sundays and public holidays.",
        "urdu": "ہم پیر تا جمعہ صبح 9 سے شام 5 بجے تک اور ہفتے کو صبح 10 سے دوپہر 2 بجے تک کھلے ہیں۔ اتوار اور عوامی چھٹیوں پر بند ہیں۔",
        "hindi": "हम सोमवार से शुक्रवार सुबह 9 से शाम 5 बजे तक और शनिवार को सुबह 10 से दोपहर 2 बजे तक खुले हैं। रविवार और सार्वजनिक छुट्टियों पर बंद हैं।",
        "spanish": "Estamos abiertos de lunes a viernes de 9 a.m. a 5 p.m., y los sábados de 10 a.m. a 2 p.m. Cerramos los domingos y días festivos.",
    },
    "ask_price": {
        "english": "A general consultation is $50, a dental checkup is $40, a blood test is $30, vaccination is $25, physiotherapy is $60, and an X-ray is $80.",
        "urdu": "عمومی معائنہ $50، دانتوں کا چیک اپ $40، خون کا ٹیسٹ $30، ویکسینیشن $25، فیزیوتھراپی $60 اور ایکس رے $80 ہے۔",
        "hindi": "सामान्य परामर्श $50, दाँतों की जाँच $40, खून का टेस्ट $30, टीकाकरण $25, फिजियोथेरेपी $60 और एक्स-रे $80 है।",
        "spanish": "Una consulta general cuesta $50, una revisión dental $40, un análisis de sangre $30, la vacunación $25, fisioterapia $60 y una radiografía $80.",
    },
    "ask_services": {
        "english": "We offer general consultations, dental checkups, blood tests, vaccinations, physiotherapy, and X-rays.",
        "urdu": "ہم عمومی معائنہ، دانتوں کا چیک اپ، خون کے ٹیسٹ، ویکسینیشن، فیزیوتھراپی اور ایکس رے کی سہولت فراہم کرتے ہیں۔",
        "hindi": "हम सामान्य परामर्श, दाँतों की जाँच, खून के टेस्ट, टीकाकरण, फिजियोथेरेपी और एक्स-रे की सुविधा प्रदान करते हैं।",
        "spanish": "Ofrecemos consultas generales, revisiones dentales, análisis de sangre, vacunación, fisioterapia y radiografías.",
    },
    "ask_location": {
        "english": "We are located at 123 Demo Street, Springfield. Street parking is available nearby.",
        "urdu": "ہم 123 ڈیمو اسٹریٹ، سپرنگ فیلڈ میں واقع ہیں۔ پاس میں سٹریٹ پارکنگ دستیاب ہے۔",
        "hindi": "हम 123 डेमो स्ट्रीट, स्प्रिंगफील्ड में स्थित हैं। पास में स्ट्रीट पार्किंग उपलब्ध है।",
        "spanish": "Estamos ubicados en la Calle Demo 123, Springfield. Hay estacionamiento en la calle cerca.",
    },
    "ask_human": {
        "english": "I understand. I can connect you to a human representative. Please hold while I transfer your call.",
        "urdu": "میں سمجھتا ہوں۔ میں آپ کو کسی انسانی نمائندے سے جوڑ سکتا ہوں۔ براہ کرم کال ٹرانسفر ہونے تک انتظار کریں۔",
        "hindi": "मैं समझता हूँ। मैं आपको किसी मानव प्रतिनिधि से जोड़ सकता हूँ। कृपया कॉल ट्रांसफर होने तक प्रतीक्षा करें।",
        "spanish": "Entiendo. Puedo conectarle con un representante humano. Espere mientras transfiero su llamada.",
    },
    "ask_identity": {
        "english": "I am an AI assistant, not a human. If you would like, I can connect you to a human representative.",
        "urdu": "میں ایک AI اسسٹنٹ ہوں، انسان نہیں۔ اگر آپ چاہیں تو میں آپ کو کسی انسانی نمائندے سے جوڑ سکتا ہوں۔",
        "hindi": "मैं एक AI सहायक हूँ, इंसान नहीं। यदि आप चाहें तो मैं आपको किसी मानव प्रतिनिधि से जोड़ सकता हूँ।",
        "spanish": "Soy un asistente de IA, no un humano. Si lo desea, puedo conectarle con un representante humano.",
    },
    "goodbye": {
        "english": "Thank you for calling Sunrise Family Clinic. Goodbye!",
        "urdu": "سنرے فیملی کلینک پر کال کرنے کا شکریہ۔ الوداع!",
        "hindi": "सनराइज़ फैमिली क्लिनिक पर कॉल करने के लिए धन्यवाद। अलविदा!",
        "spanish": "Gracias por llamar a la Clínica Familiar Sunrise. ¡Adiós!",
    },
    "out_of_scope": {
        "english": "I am not sure about that. I will pass your request to a human representative who can help. Would you like me to do that?",
        "urdu": "مجھے اس بارے میں یقین نہیں۔ میں آپ کی درخواست کسی انسانی نمائندے کو دوں گا جو مدد کر سکتا ہے۔ کیا آپ چاہتے ہیں کہ میں ایسا کروں؟",
        "hindi": "मुझे इस बारे में यकीन नहीं है। मैं आपका अनुरोध किसी मानव प्रतिनिधि को दूँगा जो मदद कर सकता है। क्या आप चाहते हैं कि मैं ऐसा करूँ?",
        "spanish": "No estoy seguro de eso. Pasaré su solicitud a un representante humano que pueda ayudar. ¿Quiere que lo haga?",
    },
    "not_offered": {
        "english": "I am sorry, but we do not offer that service. I can pass your request to a human representative if you like.",
        "urdu": "معذرت، ہم یہ خدمت نہیں دیتے۔ اگر آپ چاہیں تو میں آپ کی درخواست کسی انسانی نمائندے کو دے سکتا ہوں۔",
        "hindi": "क्षमा करें, हम यह सेवा नहीं देते। यदि आप चाहें तो मैं आपका अनुरोध किसी मानव प्रतिनिधि को दे सकता हूँ।",
        "spanish": "Lo siento, pero no ofrecemos ese servicio. Puedo pasar su solicitud a un representante humano si lo desea.",
    },
    "stop": {
        "english": "Okay, I will stop. Please go ahead.",
        "urdu": "ٹھیک ہے، میں رک جاؤں گا۔ براہ کرم بتائیں۔",
        "hindi": "ठीक है, मैं रूक जाऊँगा। कृपया बताइए।",
        "spanish": "Está bien, me detendré. Continúe, por favor.",
    },
}

# Booking prompts per language, per missing field.
BOOKING_PROMPTS = {
    "name": {
        "english": "May I have your name, please?",
        "urdu": "براہ کرم اپنا نام بتائیں؟",
        "hindi": "कृपया अपना नाम बताइए?",
        "spanish": "¿Me puede dar su nombre, por favor?",
    },
    "day": {
        "english": "Which day would you like to come in? We are open Monday to Saturday.",
        "urdu": "آپ کس دن آنا چاہتے ہیں؟ ہم پیر تا ہفتہ کھلے ہیں۔",
        "hindi": "आप किस दिन आना चाहते हैं? हम सोमवार से शनिवार खुले हैं।",
        "spanish": "¿Qué día le gustaría venir? Estamos abiertos de lunes a sábado.",
    },
    "time": {
        "english": "What time would you prefer? We have slots from 9 AM to 4 PM.",
        "urdu": "آپ کون سا وقت پسند کریں گے؟ ہمارے پاس صبح 9 سے شام 4 بجے تک سلاٹس ہیں۔",
        "hindi": "आप कौन सा समय पसंद करेंगे? हमारे पास सुबह 9 से शाम 4 बजे तक स्लॉट हैं।",
        "spanish": "¿Qué hora prefiere? Tenemos horarios de 9 a.m. a 4 p.m.",
    },
    "confirm": {
        "english": "Let me confirm: an appointment for {name} on {day} at {time}. Is that correct?",
        "urdu": "تصدیق کرتا ہوں: {name} کے لیے {day} کو {time} بجے ایپائنٹمنٹ۔ کیا یہ درست ہے؟",
        "hindi": "पुष्टि करता हूँ: {name} के लिए {day} को {time} बजे अपॉइंटमेंट। क्या यह सही है?",
        "spanish": "Permítame confirmar: una cita para {name} el {day} a las {time}. ¿Es correcto?",
    },
    "done": {
        "english": "Your appointment is booked for {day} at {time}. We look forward to seeing you. Goodbye!",
        "urdu": "آپ کی ایپائنٹمنٹ {day} کو {time} بجے بک ہو گئی ہے۔ آپ سے ملنے کی امید ہے۔ الوداع!",
        "hindi": "आपकी अपॉइंटमेंट {day} को {time} बजे बुक हो गई है। आपसे मिलने की उम्मीद है। अलविदा!",
        "spanish": "Su cita está reservada para el {day} a las {time}. Esperamos verle. ¡Adiós!",
    },
}

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
DAY_ALIASES = {
    "monday": "monday", "mon": "monday", "پیر": "monday", "सोम": "monday", "som": "monday", "lunes": "monday",
    "tuesday": "tuesday", "tue": "tuesday", "منگل": "tuesday", "मंगल": "tuesday", "tuesday": "tuesday", "martes": "tuesday",
    "wednesday": "wednesday", "wed": "wednesday", "بدھ": "wednesday", "बुध": "wednesday", "budh": "wednesday", "miércoles": "wednesday", "miercoles": "wednesday",
    "thursday": "thursday", "thu": "thursday", "جمعرات": "thursday", "गुरु": "thursday", "guru": "thursday", "jueves": "thursday",
    "friday": "friday", "fri": "friday", "جمعہ": "friday", "शुक्र": "friday", "shukr": "friday", "viernes": "friday",
    "saturday": "saturday", "sat": "saturday", "ہفتہ": "saturday", "शनि": "saturday", "shani": "saturday", "sábado": "saturday", "sabado": "saturday",
    "sunday": "sunday", "اتوار": "sunday", "रवि": "sunday", "ravi": "sunday", "domingo": "sunday",
}

TIMES = ["9:00 am", "10:00 am", "11:00 am", "12:00 pm", "1:00 pm", "2:00 pm", "3:00 pm", "4:00 pm"]
TIME_ALIASES = {
    "9": "9:00 am", "9am": "9:00 am", "9:00": "9:00 am", "صبح 9": "9:00 am", "सुबह 9": "9:00 am",
    "10": "10:00 am", "10am": "10:00 am", "10:00": "10:00 am", "صبح 10": "10:00 am", "सुबह 10": "10:00 am",
    "11": "11:00 am", "11am": "11:00 am", "11:00": "11:00 am", "صبح 11": "11:00 am", "सुबह 11": "11:00 am",
    "12": "12:00 pm", "12pm": "12:00 pm", "12:00": "12:00 pm", "دوپہر 12": "12:00 pm", "दोपहर 12": "12:00 pm",
    "1": "1:00 pm", "1pm": "1:00 pm", "1:00": "1:00 pm", "1 بجے": "1:00 pm", "1 बजे": "1:00 pm",
    "2": "2:00 pm", "2pm": "2:00 pm", "2:00": "2:00 pm", "2 بجے": "2:00 pm", "2 बजे": "2:00 pm",
    "3": "3:00 pm", "3pm": "3:00 pm", "3:00": "3:00 pm", "3 بجے": "3:00 pm", "3 बजे": "3:00 pm",
    "4": "4:00 pm", "4pm": "4:00 pm", "4:00": "4:00 pm", "4 بجے": "4:00 pm", "4 बजे": "4:00 pm",
}


def _extract_day(text):
    t = text.lower()
    for alias, day in DAY_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", t):
            return day
    return None


def _extract_time(text):
    t = text.lower()
    for alias, tm in TIME_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", t):
            return tm
    return None


def _extract_name(text):
    """Extract a name after 'name is' / 'my name is' / 'میرا نام' / 'मेरा नाम'."""
    m = re.search(r"(?:my name is|name is|میرا نام|میرا نام ہے|मेरा नाम|मेरा नाम है|me llamo|mi nombre es)\s+([a-zA-Z\u0600-\u06FF\u0900-\u097F]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


# ------------------------------------------------------------------ brain

class BrainResponse:
    def __init__(self, text, language, intent, action="speak", state=None):
        self.text = text
        self.language = language
        self.intent = intent
        self.action = action  # speak | stop | handoff | end
        self.state = state

    def __repr__(self):
        return f"BrainResponse(lang={self.language}, intent={self.intent}, action={self.action}, text={self.text!r})"


class Brain:
    """State machine for the clinic call. Text in, reply text out."""

    def __init__(self, scenario_path="scenarios/clinic.yaml", use_llm=False):
        with open(scenario_path, encoding="utf-8") as f:
            self.scenario = yaml.safe_load(f)
        self.use_llm = use_llm
        self.llm = None
        self.reset()

    def reset(self):
        self.state = "greet"
        self.language = "english"
        self.history = []          # list of caller utterances
        self.booking = {}          # name, day, time
        self.greeted = False
        self.pending_handoff = False

    # ------------------------------------------------------------- LLM (optional)
    def _get_llm(self):
        if not self.use_llm:
            return None
        if self.llm is None:
            try:
                from transformers import pipeline
                self.llm = pipeline("text-generation",
                                     model="Qwen/Qwen2.5-1.5B-Instruct",
                                     device_map="auto")
            except Exception:
                self.llm = None
        return self.llm

    def _llm_classify(self, text):
        llm = self._get_llm()
        if llm is None:
            return None
        intents = list(self.scenario["intents"])
        prompt = (f"Classify this caller utterance into exactly one intent: {intents}.\n"
                  f"Utterance: {text}\nIntent:")
        try:
            out = llm(prompt, max_new_tokens=10, do_sample=False)
            pred = out[0]["generated_text"].split("Intent:")[-1].strip().lower()
            for intent in intents:
                if intent in pred:
                    return intent
        except Exception:
            return None
        return None

    def _llm_rephrase(self, text, language):
        """Lightly rephrase an approved answer. Falls back to the original."""
        llm = self._get_llm()
        if llm is None:
            return text
        prompt = (f"Rephrase the following customer-service reply in {language}, "
                  f"keeping it short and natural. Reply with only the rephrased sentence.\n"
                  f"Reply: {text}\nRephrased:")
        try:
            out = llm(prompt, max_new_tokens=60, do_sample=False)
            pred = out[0]["generated_text"].split("Rephrased:")[-1].strip()
            return pred if pred else text
        except Exception:
            return text

    # ------------------------------------------------------------- helpers
    def _answer(self, intent, language):
        table = ANSWERS.get(intent, ANSWERS["out_of_scope"])
        return table.get(language, table["english"])

    def _set_language(self, text):
        # Follow the caller's current language on every utterance (they may
        # switch back and forth). detect_language is script-decisive for
        # Urdu/Hindi and stopword-based for English/Spanish.
        self.language = detect_language(text)

    # ------------------------------------------------------------- main
    def process(self, text):
        """Process one caller utterance. Returns a BrainResponse."""
        text = (text or "").strip()
        if not text:
            return BrainResponse("", self.language, "empty", "speak", self.state)

        self.history.append(text)
        self._set_language(text)
        lang = self.language

        # interruption / stop
        intent, _kw = classify_intent(text)
        llm_intent = self._llm_classify(text) if self.use_llm else None
        if llm_intent:
            intent = llm_intent

        # --- greeting state
        if self.state == "greet":
            self.greeted = True
            self.state = "understand"
            return BrainResponse(self._answer("greeting", lang), lang, "greeting",
                                 "speak", self.state)

        # --- explicit stop
        if intent == "stop":
            return BrainResponse(self._answer("stop", lang), lang, "stop",
                                 "stop", self.state)

        # --- goodbye
        if intent == "goodbye":
            self.state = "end"
            return BrainResponse(self._answer("goodbye", lang), lang, "goodbye",
                                 "end", self.state)

        # --- identity / human
        if intent == "ask_identity":
            return BrainResponse(self._answer("ask_identity", lang), lang,
                                 "ask_identity", "speak", self.state)
        if intent == "ask_human":
            self.pending_handoff = True
            return BrainResponse(self._answer("ask_human", lang), lang,
                                 "ask_human", "handoff", self.state)

        # --- booking flow
        if self.state == "booking":
            return self._handle_booking(text, lang)

        if intent == "book_appointment":
            self.state = "booking"
            return self._handle_booking(text, lang, start=True)

        # --- simple FAQ intents
        if intent in ANSWERS:
            return BrainResponse(self._answer(intent, lang), lang, intent,
                                 "speak", self.state)

        # --- cancel / reschedule: offer human (we have no scheduling system)
        if intent in ("cancel_appointment", "reschedule_appointment"):
            self.pending_handoff = True
            return BrainResponse(self._answer("ask_human", lang), lang,
                                 "ask_human", "handoff", self.state)

        # --- out of scope / not offered
        return BrainResponse(self._answer("out_of_scope", lang), lang,
                             "out_of_scope", "speak", self.state)

    def _handle_booking(self, text, lang, start=False):
        # collect fields in order
        if "name" not in self.booking:
            name = _extract_name(text)
            if name:
                self.booking["name"] = name
            else:
                return BrainResponse(BOOKING_PROMPTS["name"][lang], lang,
                                     "book_appointment", "speak", self.state)
        if "day" not in self.booking:
            day = _extract_day(text)
            if day:
                self.booking["day"] = day
            else:
                return BrainResponse(BOOKING_PROMPTS["day"][lang], lang,
                                     "book_appointment", "speak", self.state)
        if "time" not in self.booking:
            tm = _extract_time(text)
            if tm:
                self.booking["time"] = tm
            else:
                return BrainResponse(BOOKING_PROMPTS["time"][lang], lang,
                                     "book_appointment", "speak", self.state)
        # all collected -> confirm
        if "confirmed" not in self.booking:
            self.booking["confirmed"] = False
            prompt = BOOKING_PROMPTS["confirm"][lang].format(**self.booking)
            return BrainResponse(prompt, lang, "book_appointment", "speak",
                                 self.state)
        # caller confirmed (yes/haan/ha/yes)
        if re.search(r"\b(yes|yeah|haan|ha|हाँ|हاں|ji|جی ہاں|sí|si|ok|okay|theek|ٹھیک)\b", text.lower()):
            self.booking["confirmed"] = True
            self.state = "end"
            done = BOOKING_PROMPTS["done"][lang].format(**self.booking)
            return BrainResponse(done, lang, "book_appointment", "end", self.state)
        # caller corrected something -> restart booking
        self.booking = {}
        return self._handle_booking(text, lang, start=True)
