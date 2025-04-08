from textblob import TextBlob
from langdetect import detect, detect_langs
from typing import Dict, Tuple, List
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_threshold = -0.2  # Threshold for negative sentiment
        self.confidence_threshold = 0.8  # Threshold for language detection confidence
        
    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment of text"""
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity
        
        # Categorize sentiment
        if sentiment_score > 0.2:
            sentiment = "positive"
        elif sentiment_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return sentiment_score, sentiment
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language and confidence"""
        try:
            # Get language probabilities
            lang_probabilities = detect_langs(text)
            
            # Get the most likely language and its probability
            best_lang = max(lang_probabilities, key=lambda x: x.prob)
            return best_lang.lang, best_lang.prob
            
        except Exception:
            # Fallback to simple detection if detailed detection fails
            try:
                lang = detect(text)
                return lang, 1.0
            except:
                return "unknown", 0.0
    
    def should_redirect_to_human(self, text: str) -> Tuple[bool, Dict[str, any]]:
        """Determine if the query should be redirected to human support"""
        sentiment_score, sentiment = self.analyze_sentiment(text)
        language, confidence = self.detect_language(text)
        
        # Reasons for redirection
        reasons = []
        
        # Check sentiment
        if sentiment_score < self.sentiment_threshold:
            reasons.append("negative_sentiment")
            
        # Check language
        if language != "en" or confidence < self.confidence_threshold:
            reasons.append("non_english")
            
        should_redirect = len(reasons) > 0
        
        return should_redirect, {
            "sentiment_score": sentiment_score,
            "sentiment": sentiment,
            "language": language,
            "confidence": confidence,
            "reasons": reasons
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """Analyze multiple texts"""
        results = []
        for text in texts:
            should_redirect, analysis = self.should_redirect_to_human(text)
            analysis["should_redirect"] = should_redirect
            results.append(analysis)
        return results 