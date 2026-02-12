import re
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

class AncientToSimpleTamil:
    """
    Module 4: Ancient Tamil to Simple Tamil Translation
    - Word-by-word meanings
    - Grammar explanations
    - Simple, conversational output
    """
    
    def __init__(self):
        # Ancient to Modern Tamil word dictionary
        self.ancient_word_dict = {
            'ஆயிரம்': {'modern': 'ஆயிரம்', 'simple': 'ஆயிரம் (1000)', 'root': 'ஆயிரம்'},
            'யாதும்': {'modern': 'எல்லா', 'simple': 'எல்லா ஊர்களும்', 'root': 'யா-தும்'},
            'ஊரே': {'modern': 'ஊர்கள்', 'simple': 'எல்லா ஊர்களும்', 'root': 'ஊர்-ஏ'},
            'யாவரும்': {'modern': 'எல்லோரும்', 'simple': 'எல்லா மனிதர்களும்', 'root': 'யா-வர்-உம்'},
            'கேளிர்': {'modern': 'உறவினர்கள்', 'simple': 'நம் உறவினர்கள்', 'root': 'கேள்-இர்'},
            'தீதும்': {'modern': 'தீமை', 'simple': 'கெட்டது', 'root': 'தீது-உம்'},
            'நன்றும்': {'modern': 'நன்மை', 'simple': 'நல்லது', 'root': 'நன்று-உம்'},
            'பிறர்தர': {'modern': 'பிறர் தருவது', 'simple': 'மற்றவர்கள் கொடுப்பது', 'root': 'பிறர்-தர'},
            'வாரா': {'modern': 'வராது', 'simple': 'வருவதில்லை', 'root': 'வா-ஆ'},
            'அறம்': {'modern': 'அறம்', 'simple': 'நல்லொழுக்கம்', 'root': 'அறம்'},
            'பொருள்': {'modern': 'பொருள்', 'simple': 'செல்வம், பொருள்', 'root': 'பொருள்'},
            'இன்பம்': {'modern': 'இன்பம்', 'simple': 'மகிழ்ச்சி', 'root': 'இன்பம்'},
            'வீடு': {'modern': 'வீடு', 'simple': 'வீடு, விடுதலை', 'root': 'வீடு'},
        }
        
        # Grammar patterns
        self.grammar_patterns = [
            (r'யாதும்', 'எல்லா'),
            (r'ஊரே', 'ஊர்கள்'),
            (r'யாவரும்', 'எல்லோரும்'),
            (r'கேளிர்', 'உறவினர்கள்'),
            (r'தீதும்', 'தீமை'),
            (r'நன்றும்', 'நன்மை'),
        ]
        
        # Load mT5 model (optional)
        try:
            self.tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
            self.model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
            self.use_mt5 = True
        except:
            self.use_mt5 = False
            print("mT5 not available, using rule-based translation")
    
    def translate(self, modern_text, detailed_meanings=True):
        """
        Translate modern Tamil text to simple, conversational Tamil
        """
        if self.use_mt5:
            return self.mt5_translate(modern_text, detailed_meanings)
        else:
            return self.rule_based_translate(modern_text, detailed_meanings)
    
    def rule_based_translate(self, modern_text, detailed_meanings=True):
        """
        Rule-based translation with dictionary lookup
        """
        # Split into words
        words = modern_text.split()
        
        # Simple translation
        simple_words = []
        word_meanings = {}
        grammar_notes = []
        
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[.,!?;:]', '', word)
            
            if clean_word in self.ancient_word_dict:
                # Dictionary match
                simple_words.append(self.ancient_word_dict[clean_word]['modern'])
                if detailed_meanings:
                    word_meanings[clean_word] = self.ancient_word_dict[clean_word]
                
                # Add grammar note for specific patterns
                if clean_word == 'யாதும்':
                    grammar_notes.append("'யாதும்' என்பது 'எல்லா' எனப் பொருள்படும்")
                elif clean_word == 'கேளிர்':
                    grammar_notes.append("'கேளிர்' என்பது 'உறவினர்கள்' எனப் பொருள்படும்")
            else:
                simple_words.append(clean_word)
        
        # Apply grammar patterns
        simple_text = ' '.join(simple_words)
        
        for pattern, replacement in self.grammar_patterns:
            simple_text = re.sub(pattern, replacement, simple_text)
        
        # Additional grammar corrections
        simple_text = self.colloquial_conversion(simple_text)
        
        # Add explanations
        if detailed_meanings and len(grammar_notes) > 0:
            simple_text += "\n\n📝 விளக்கம்:\n" + "\n".join(grammar_notes)
        
        return simple_text, word_meanings, grammar_notes
    
    def colloquial_conversion(self, text):
        """
        Convert formal Tamil to conversational Tamil
        """
        # Formal to colloquial mappings
        mappings = [
            (r'செல்கிறேன்', 'போறேன்'),
            (r'வருகிறேன்', 'வர்றேன்'),
            (r'செய்கிறேன்', 'பண்றேன்'),
            (r'கொடுக்கிறேன்', 'தர்றேன்'),
            (r'எடுக்கிறேன்', 'எடுக்கறேன்'),
            (r'பார்க்கிறேன்', 'பாக்கறேன்'),
            (r'இருக்கிறது', 'இருக்கு'),
            (r'வருகிறது', 'வருது'),
            (r'செய்கிறது', 'பண்ணுது'),
        ]
        
        for formal, colloquial in mappings:
            text = re.sub(formal, colloquial, text)
        
        return text
    
    def mt5_translate(self, modern_text, detailed_meanings=True):
        """
        Use mT5 for translation (when available)
        """
        if not self.use_mt5:
            return self.rule_based_translate(modern_text, detailed_meanings)
        
        # Prepare prompt
        prompt = f"translate Ancient Tamil to Simple Tamil: {modern_text}"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            temperature=0.7
        )
        
        # Decode
        simple_tamil = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Still provide word meanings
        words = modern_text.split()
        word_meanings = {}
        for word in words[:5]:  # Limit to first 5 words
            if word in self.ancient_word_dict:
                word_meanings[word] = self.ancient_word_dict[word]
        
        return simple_tamil, word_meanings, []
