import spacy
import nltk
from textblob import TextBlob
from gensim import corpora
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List, Any, Optional
import torch
import re
import random
import numpy as np
from collections import defaultdict

class HybridTextHumanizer:
    def __init__(self, use_gpu: bool = False):
        # Initialize traditional NLP tools
        self.nlp = spacy.load("en_core_web_sm")
        self._download_nltk_data()
        
        # Setup device for transformers
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        
        # Load multiple transformer models for different purposes
        self._load_transformer_models()
        
        # Initialize linguistic patterns
        self.setup_linguistic_patterns()
        self.setup_style_templates()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        from nltk.corpus import wordnet
        self.wordnet = wordnet
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
    
    def _load_transformer_models(self):
        """Load various transformer models for different humanization tasks"""
        print("Loading transformer models...")
        
        try:
            # Model 1: General paraphrasing (T5-based)
            self.paraphrase_pipeline = pipeline(
                "text2text-generation",
                model="humarin/chatgpt_paraphraser_on_T5_base",
                device=self.device,
                torch_dtype=torch.float16 if self.device >= 0 else torch.float32
            )
            
            # Model 2: Formality/style transfer
            try:
                self.formality_pipeline = pipeline(
                    "text-classification",
                    model="s-nlp/roberta-base-formality-ranker",
                    device=self.device
                )
            except:
                self.formality_pipeline = None
            
            # Model 3: Text simplification
            try:
                self.simplification_pipeline = pipeline(
                    "text2text-generation",
                    model="microsoft/t5-base-simplify",
                    device=self.device
                )
            except:
                self.simplification_pipeline = None
            
            print("Transformer models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading some models: {e}")
            # Fallback to basic models
            self.paraphrase_pipeline = pipeline(
                "text2text-generation",
                model="t5-small",
                device=self.device
            )

    def setup_linguistic_patterns(self):
        """Enhanced linguistic patterns"""
        self.contractions = {
            "cannot": "can't", "will not": "won't", "do not": "don't",
            "does not": "doesn't", "is not": "isn't", "are not": "aren't",
            "have not": "haven't", "has not": "hasn't", "had not": "hadn't"
        }
        
        self.formal_to_informal = {
            "utilize": "use", "approximately": "about", "assistance": "help",
            "commence": "start", "demonstrate": "show", "encounter": "meet",
            "facilitate": "help", "implement": "do", "inquire": "ask"
        }

    def setup_style_templates(self):
        """Enhanced style templates with transformer-specific parameters"""
        self.style_templates = {
            "casual": {
                "temperature": 0.8,
                "repetition_penalty": 1.2,
                "max_length": 256,
                "formality_target": "informal",
                "simplify": True,
                "use_contractions": True
            },
            "professional": {
                "temperature": 0.3,
                "repetition_penalty": 1.1,
                "max_length": 512,
                "formality_target": "formal",
                "simplify": False,
                "use_contractions": False
            },
            "storytelling": {
                "temperature": 0.9,
                "repetition_penalty": 1.3,
                "max_length": 512,
                "formality_target": "informal",
                "simplify": False,
                "use_contractions": True
            },
            "academic": {
                "temperature": 0.2,
                "repetition_penalty": 1.0,
                "max_length": 1024,
                "formality_target": "formal",
                "simplify": False,
                "use_contractions": False
            }
        }

    def humanize_text(self, text: str, style: str = "casual") -> Dict[str, Any]:
        """
        Enhanced humanization using hybrid approach
        """
        original_text = text
        
        # Step 1: Pre-process with traditional NLP
        pre_processed = self.pre_process_text(text)
        
        # Step 2: Analyze original text
        original_analysis = self.analyze_text(text)
        
        # Step 3: Apply transformer-based humanization
        style_config = self.style_templates.get(style, self.style_templates["casual"])
        humanized_text = self.transformer_humanize(pre_processed, style_config)
        
        # Step 4: Post-process with traditional NLP
        final_text = self.post_process_text(humanized_text, style_config)
        
        # Step 5: Analyze final text
        final_analysis = self.analyze_text(final_text)
        
        return {
            "original_text": original_text,
            "humanized_text": final_text,
            "metrics": {
                "readability_change": final_analysis["readability"] - original_analysis["readability"],
                "sentence_variation_change": final_analysis["sentence_variation"] - original_analysis["sentence_variation"],
                "formality_change": final_analysis["formality"] - original_analysis["formality"],
                "original_analysis": original_analysis,
                "humanized_analysis": final_analysis,
                "method_used": "hybrid_transformer_nlp"
            }
        }

    def pre_process_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Pre-process text using traditional NLP tools
        Returns structured data for transformer processing
        """
        doc = self.nlp(text)
        processed_chunks = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            # Analyze with TextBlob
            blob = TextBlob(sent_text)
            
            # Get semantic features with Gensim
            tokens = [token.lemma_.lower() for token in sent if token.is_alpha]
            
            processed_chunks.append({
                "text": sent_text,
                "sentiment": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity,
                "tokens": tokens,
                "length": len(sent_text),
                "word_count": len([token for token in sent if token.is_alpha])
            })
        
        return processed_chunks

    def transformer_humanize(self, chunks: List[Dict[str, Any]], style_config: Dict) -> str:
        """
        Core humanization using transformer models
        """
        humanized_sentences = []
        
        for chunk in chunks:
            original_text = chunk["text"]
            
            # Build context-aware prompt
            prompt = self._build_transformer_prompt(original_text, style_config, chunk)
            
            try:
                # Use paraphrase model as primary humanizer
                paraphrased = self.paraphrase_pipeline(
                    prompt,
                    max_length=style_config["max_length"],
                    temperature=style_config["temperature"],
                    repetition_penalty=style_config["repetition_penalty"],
                    num_return_sequences=1,
                    do_sample=True
                )[0]['generated_text']
                
                # Apply simplification if needed
                if (style_config.get("simplify", False) and 
                    self.simplification_pipeline and 
                    len(original_text.split()) > 15):
                    try:
                        simplified = self.simplification_pipeline(
                            f"simplify: {paraphrased}",
                            max_length=style_config["max_length"],
                            num_return_sequences=1
                        )[0]['generated_text']
                        humanized_sentences.append(simplified)
                    except:
                        humanized_sentences.append(paraphrased)
                else:
                    humanized_sentences.append(paraphrased)
                    
            except Exception as e:
                print(f"Transformer humanization failed: {e}")
                # Fallback to traditional method
                humanized_sentences.append(self._traditional_humanize_fallback(original_text, style_config))
        
        return " ".join(humanized_sentences)

    def _build_transformer_prompt(self, text: str, style_config: Dict, chunk: Dict) -> str:
        """Build sophisticated prompts for transformer models"""
        
        style_prompts = {
            "casual": f"Paraphrase this to sound like casual human conversation. Use contractions and informal language: {text}",
            "professional": f"Rephrase this professionally while keeping it human and approachable. Avoid robotic language: {text}",
            "storytelling": f"Rewrite this as engaging storytelling. Use vivid language and narrative flow: {text}",
            "academic": f"Rephrase this academically while maintaining readability and human tone: {text}"
        }
        
        base_prompt = style_prompts.get(style_config.get("formality_target", "casual"), style_prompts["casual"])
        
        # Add sentiment guidance
        sentiment = chunk.get("sentiment", 0)
        if abs(sentiment) > 0.3:
            emotion = "positive" if sentiment > 0 else "negative"
            base_prompt += f" Use a slightly {emotion} tone."
        
        # Add length guidance
        if chunk.get("word_count", 0) > 25:
            base_prompt += " Make it more concise."
        
        return base_prompt

    def _traditional_humanize_fallback(self, text: str, style_config: Dict) -> str:
        """Fallback humanization using traditional NLP methods"""
        # Apply contractions for informal styles
        if style_config.get("use_contractions", False):
            for formal, informal in self.contractions.items():
                text = re.sub(r'\b' + formal + r'\b', informal, text, flags=re.IGNORECASE)
        
        # Adjust vocabulary formality
        if style_config.get("formality_target") == "informal":
            for formal, informal in self.formal_to_informal.items():
                text = re.sub(r'\b' + formal + r'\b', informal, text, flags=re.IGNORECASE)
        
        return text

    def post_process_text(self, text: str, style_config: Dict) -> str:
        """
        Post-process transformer output with traditional NLP
        """
        # Fix common transformer artifacts
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Ensure proper sentence casing
        sentences = nltk.sent_tokenize(text)
        sentences = [sentence[0].upper() + sentence[1:] if sentence else "" 
                    for sentence in sentences]
        
        # Final style-specific adjustments
        if style_config.get("use_contractions", False):
            sentences = [self._enhance_informality(sent) for sent in sentences]
        
        return " ".join(sentences)

    def _enhance_informality(self, sentence: str) -> str:
        """Add informal touches using traditional NLP"""
        # Add conversational elements occasionally
        if random.random() < 0.3 and len(sentence.split()) > 5:
            informal_starters = ["You know,", "Well,", "Actually,", "I mean,"]
            sentence = f"{random.choice(informal_starters)} {sentence.lower()}"
        
        return sentence

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Enhanced text analysis using both traditional and transformer methods"""
        # Traditional analysis
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        word_count = len([token for token in doc if not token.is_punct])
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Sentence variation
        sentence_lengths = [len([token for token in sent if not token.is_punct]) for sent in sentences]
        sentence_variation = np.std(sentence_lengths) if sentence_lengths else 0
        
        # Readability
        readability = self.calculate_readability(text)
        
        # Formality analysis using transformer or fallback
        formality = self.analyze_formality(text)
        
        # Sentiment analysis
        sentiment = TextBlob(text).sentiment.polarity
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "sentence_variation": sentence_variation,
            "readability": readability,
            "formality": formality,
            "sentiment": sentiment
        }

    def analyze_formality(self, text: str) -> float:
        """Analyze formality using transformer model or fallback"""
        if self.formality_pipeline:
            try:
                result = self.formality_pipeline(text[:512])[0]  # Truncate for model limits
                # Convert formality score to 0-1 scale
                if result['label'] == 'formal':
                    return result['score']
                else:
                    return 1 - result['score']
            except:
                pass
        # Fallback to traditional formality analysis
        return self._traditional_formality_analysis(text)

    def _traditional_formality_analysis(self, text: str) -> float:
        """Traditional formality analysis as fallback"""
        doc = self.nlp(text)
        formal_indicators = 0
        total_words = 0
        
        for token in doc:
            if token.is_alpha and not token.is_stop:
                total_words += 1
                if token.lemma_.lower() in self.formal_to_informal:
                    formal_indicators += 1
                if token.tag_ in ["NN", "NNS"] and len(token.text) > 8:
                    formal_indicators += 0.5
        
        return formal_indicators / total_words if total_words > 0 else 0.5

    def calculate_readability(self, text: str) -> float:
        """Calculate readability score"""
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        complex_words = [word for word in words if len(word) > 6]
        
        # Simple readability heuristic
        readability = 100 - (avg_sentence_length + (len(complex_words) / len(words) * 100))
        return max(0, min(100, readability))