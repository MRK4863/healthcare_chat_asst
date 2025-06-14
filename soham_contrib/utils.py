import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
from itertools import combinations
import re

class MedicalRAGIndexer:
    def __init__(self):
        # Core data structures
        self.disease_symptoms = {}          # Disease -> {symptoms: list, count: int, frequency: dict}
        self.symptom_diseases = {}          # Symptom -> {diseases: list, frequency: dict}
        self.symptom_combinations = {}      # (symptom1, symptom2) -> {diseases: list, scores: dict}
        self.disease_profiles = {}          # Disease -> complete profile
        
        # Vector representations
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
        self.disease_vectors = None
        self.symptom_vectors = None
        
        # Frequency and scoring data
        self.global_symptom_freq = Counter()
        self.disease_symptom_count = {}
        
        # Processed data
        self.processed_df = None
        
    def load_and_process_csv(self, csv_path):
        """Load CSV and perform initial processing"""
        print("Loading CSV file...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} diseases")
        
        # Process the dataset
        self.processed_df = self._aggregate_dataset(df)
        print(f"Processed {len(self.processed_df)} unique diseases")
        
        return self.processed_df
    
    def _aggregate_dataset(self, df):
        """Aggregate the wide-format dataset into structured format"""
        print("Aggregating dataset...")
        
        processed_data = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing row {idx}...")
                
            disease = str(row['Disease']).strip().lower()
            
            # Extract all non-null symptoms
            symptoms = []
            for i in range(1, 18):  # Symptom_1 to Symptom_17
                symptom_col = f'Symptom_{i}'
                if symptom_col in row and pd.notna(row[symptom_col]):
                    symptom = str(row[symptom_col]).strip().lower()
                    if symptom and symptom != 'nan':
                        symptoms.append(symptom)
            
            if disease and symptoms:  # Only add if we have both disease and symptoms
                processed_data.append({
                    'disease': disease,
                    'symptoms': symptoms,
                    'symptom_count': len(symptoms)
                })
        
        return pd.DataFrame(processed_data)
    
    def create_indexes(self):
        """Create all necessary indexes for RAG retrieval"""
        print("Creating indexes...")
        
        self._create_disease_symptom_index()
        self._create_symptom_disease_index()
        self._create_symptom_combination_index()
        self._create_vector_representations()
        self._calculate_frequencies_and_weights()
        
        print("All indexes created successfully!")
    
    def _create_disease_symptom_index(self):
        """Create disease -> symptoms mapping with frequencies"""
        print("Creating disease-symptom index...")
        
        for _, row in self.processed_df.iterrows():
            disease = row['disease']
            symptoms = row['symptoms']
            
            if disease not in self.disease_symptoms:
                self.disease_symptoms[disease] = {
                    'symptoms': set(),
                    'symptom_list': [],
                    'count': 0
                }
            
            # Add symptoms to the disease
            self.disease_symptoms[disease]['symptoms'].update(symptoms)
            self.disease_symptoms[disease]['symptom_list'].extend(symptoms)
            self.disease_symptoms[disease]['count'] += len(symptoms)
            
        # Convert sets to lists and calculate frequencies
        for disease in self.disease_symptoms:
            self.disease_symptoms[disease]['symptoms'] = list(self.disease_symptoms[disease]['symptoms'])
            symptom_freq = Counter(self.disease_symptoms[disease]['symptom_list'])
            self.disease_symptoms[disease]['frequency'] = dict(symptom_freq)
            self.disease_symptom_count[disease] = len(self.disease_symptoms[disease]['symptoms'])
    
    def _create_symptom_disease_index(self):
        """Create symptom -> diseases mapping"""
        print("Creating symptom-disease index...")
        
        for disease, data in self.disease_symptoms.items():
            for symptom in data['symptoms']:
                if symptom not in self.symptom_diseases:
                    self.symptom_diseases[symptom] = {
                        'diseases': [],
                        'frequency': Counter()
                    }
                
                self.symptom_diseases[symptom]['diseases'].append(disease)
                self.symptom_diseases[symptom]['frequency'][disease] += data['frequency'].get(symptom, 1)
        
        # Update global symptom frequency
        for symptom in self.symptom_diseases:
            self.global_symptom_freq[symptom] = len(self.symptom_diseases[symptom]['diseases'])
    
    def _create_symptom_combination_index(self):
        """Create symptom combination -> diseases mapping"""
        print("Creating symptom combination index...")
        
        for disease, data in self.disease_symptoms.items():
            symptoms = data['symptoms']
            
            # Create combinations of 2 and 3 symptoms
            for r in [2, 3]:
                if len(symptoms) >= r:
                    for combo in combinations(symptoms, r):
                        combo_key = tuple(sorted(combo))
                        
                        if combo_key not in self.symptom_combinations:
                            self.symptom_combinations[combo_key] = {
                                'diseases': [],
                                'scores': Counter()
                            }
                        
                        self.symptom_combinations[combo_key]['diseases'].append(disease)
                        # Score based on how many of the combo symptoms appear in the disease
                        combo_score = sum(data['frequency'].get(s, 1) for s in combo)
                        self.symptom_combinations[combo_key]['scores'][disease] = combo_score
    
    def _create_vector_representations(self):
        """Create TF-IDF vectors for semantic similarity"""
        print("Creating vector representations...")
        
        # Prepare documents for vectorization
        disease_documents = []
        disease_names = []
        
        for disease, data in self.disease_symptoms.items():
            # Create a document from all symptoms
            symptom_text = ' '.join(data['symptoms'])
            disease_documents.append(symptom_text)
            disease_names.append(disease)
        
        # Fit vectorizer and transform
        self.disease_vectors = self.vectorizer.fit_transform(disease_documents)
        self.disease_names = disease_names
        
        print(f"Created vectors for {len(disease_names)} diseases")
    
    def _calculate_frequencies_and_weights(self):
        """Calculate various frequency-based weights"""
        print("Calculating frequencies and weights...")
        
        total_diseases = len(self.disease_symptoms)
        
        # Calculate IDF-like weights for symptoms
        for symptom in self.symptom_diseases:
            disease_count = len(self.symptom_diseases[symptom]['diseases'])
            idf_weight = np.log(total_diseases / disease_count) if disease_count > 0 else 0
            self.symptom_diseases[symptom]['idf_weight'] = idf_weight
    
    def query_diseases(self, input_symptoms, top_k=10):
        """Main query function for RAG retrieval"""
        if not input_symptoms:
            return []
        
        # Clean and normalize input symptoms
        normalized_symptoms = [s.strip().lower() for s in input_symptoms if s.strip()]
        
        # Get disease scores using multiple methods
        exact_matches = self._get_exact_matches(normalized_symptoms)
        combination_matches = self._get_combination_matches(normalized_symptoms)
        semantic_matches = self._get_semantic_matches(normalized_symptoms)
        
        # Combine and rank results
        final_scores = self._combine_scores(exact_matches, combination_matches, semantic_matches)
        
        # Sort by score and return top_k
        ranked_diseases = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return self._format_results(ranked_diseases, normalized_symptoms)
    
    def _get_exact_matches(self, symptoms):
        """Get diseases that match exact symptoms"""
        disease_scores = Counter()
        
        for symptom in symptoms:
            if symptom in self.symptom_diseases:
                for disease in self.symptom_diseases[symptom]['diseases']:
                    # Weight by frequency and IDF
                    freq_weight = self.symptom_diseases[symptom]['frequency'][disease]
                    idf_weight = self.symptom_diseases[symptom]['idf_weight']
                    disease_scores[disease] += freq_weight * idf_weight
        
        return dict(disease_scores)
    
    def _get_combination_matches(self, symptoms):
        """Get diseases that match symptom combinations"""
        disease_scores = Counter()
        
        # Check 2-symptom combinations
        for combo in combinations(symptoms, min(2, len(symptoms))):
            combo_key = tuple(sorted(combo))
            if combo_key in self.symptom_combinations:
                for disease, score in self.symptom_combinations[combo_key]['scores'].items():
                    disease_scores[disease] += score * 1.5  # Boost combination matches
        
        # Check 3-symptom combinations if available
        if len(symptoms) >= 3:
            for combo in combinations(symptoms, 3):
                combo_key = tuple(sorted(combo))
                if combo_key in self.symptom_combinations:
                    for disease, score in self.symptom_combinations[combo_key]['scores'].items():
                        disease_scores[disease] += score * 2.0  # Higher boost for 3-symptom matches
        
        return dict(disease_scores)
    
    def _get_semantic_matches(self, symptoms):
        """Get diseases using semantic similarity"""
        if self.disease_vectors is None:
            return {}
        
        # Create query vector
        query_text = ' '.join(symptoms)
        query_vector = self.vectorizer.transform([query_text])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.disease_vectors)[0]
        
        # Convert to disease scores
        disease_scores = {}
        for i, similarity in enumerate(similarities):
            if similarity > 0.1:  # Threshold for semantic matches
                disease = self.disease_names[i]
                disease_scores[disease] = similarity * 10  # Scale up semantic scores
        
        return disease_scores
    
    def _combine_scores(self, exact, combination, semantic):
        """Combine different scoring methods"""
        all_diseases = set(exact.keys()) | set(combination.keys()) | set(semantic.keys())
        
        final_scores = {}
        for disease in all_diseases:
            exact_score = exact.get(disease, 0)
            combo_score = combination.get(disease, 0)
            semantic_score = semantic.get(disease, 0)
            
            # Weighted combination
            final_score = (exact_score * 0.5 + combo_score * 0.3 + semantic_score * 0.2)
            final_scores[disease] = final_score
        
        return final_scores
    
    def _format_results(self, ranked_diseases, input_symptoms):
        """Format results with additional information"""
        results = []
        
        for disease, score in ranked_diseases:
            if disease in self.disease_symptoms:
                disease_data = self.disease_symptoms[disease]
                
                # Calculate match statistics
                matched_symptoms = set(input_symptoms) & set(disease_data['symptoms'])
                match_ratio = len(matched_symptoms) / len(disease_data['symptoms'])
                coverage_ratio = len(matched_symptoms) / len(input_symptoms) if input_symptoms else 0
                
                result = {
                    'disease': disease,
                    'score': round(score, 4),
                    'matched_symptoms': list(matched_symptoms),
                    'all_symptoms': disease_data['symptoms'],
                    'match_ratio': round(match_ratio, 3),
                    'coverage_ratio': round(coverage_ratio, 3),
                    'total_symptoms': len(disease_data['symptoms'])
                }
                results.append(result)
        
        return results
    
    def get_symptom_suggestions(self, current_symptoms, top_diseases=5):
        """Get suggested symptoms to ask about based on current symptoms"""
        if not current_symptoms:
            return []
        
        # Get top diseases for current symptoms
        top_results = self.query_diseases(current_symptoms, top_k=top_diseases)
        
        # Collect symptoms from top diseases
        suggested_symptoms = Counter()
        current_set = set(s.lower() for s in current_symptoms)
        
        for result in top_results:
            disease_symptoms = set(result['all_symptoms'])
            # Add symptoms not yet mentioned
            new_symptoms = disease_symptoms - current_set
            for symptom in new_symptoms:
                suggested_symptoms[symptom] += result['score']
        
        # Return top suggestions
        return [symptom for symptom, _ in suggested_symptoms.most_common(10)]
    
    def save_indexes(self, filepath_prefix):
        """Save all indexes to files"""
        print(f"Saving indexes with prefix: {filepath_prefix}")
        
        # Save core data structures
        with open(f"{filepath_prefix}_disease_symptoms.pkl", 'wb') as f:
            pickle.dump(self.disease_symptoms, f)
        
        with open(f"{filepath_prefix}_symptom_diseases.pkl", 'wb') as f:
            pickle.dump(self.symptom_diseases, f)
        
        with open(f"{filepath_prefix}_symptom_combinations.pkl", 'wb') as f:
            pickle.dump(self.symptom_combinations, f)
        
        # Save vectorizer and vectors
        with open(f"{filepath_prefix}_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(f"{filepath_prefix}_disease_vectors.pkl", 'wb') as f:
            pickle.dump(self.disease_vectors, f)
        
        with open(f"{filepath_prefix}_disease_names.pkl", 'wb') as f:
            pickle.dump(self.disease_names, f)
        
        # Save frequency data
        with open(f"{filepath_prefix}_frequencies.json", 'w') as f:
            json.dump({
                'global_symptom_freq': dict(self.global_symptom_freq),
                'disease_symptom_count': self.disease_symptom_count
            }, f, indent=2)
        
        print("All indexes saved successfully!")
    
    def load_indexes(self, filepath_prefix):
        """Load all indexes from files"""
        print(f"Loading indexes with prefix: {filepath_prefix}")
        
        # Load core data structures
        with open(f"{filepath_prefix}_disease_symptoms.pkl", 'rb') as f:
            self.disease_symptoms = pickle.load(f)
        
        with open(f"{filepath_prefix}_symptom_diseases.pkl", 'rb') as f:
            self.symptom_diseases = pickle.load(f)
        
        with open(f"{filepath_prefix}_symptom_combinations.pkl", 'rb') as f:
            self.symptom_combinations = pickle.load(f)
        
        # Load vectorizer and vectors
        with open(f"{filepath_prefix}_vectorizer.pkl", 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(f"{filepath_prefix}_disease_vectors.pkl", 'rb') as f:
            self.disease_vectors = pickle.load(f)
        
        with open(f"{filepath_prefix}_disease_names.pkl", 'rb') as f:
            self.disease_names = pickle.load(f)
        
        # Load frequency data
        with open(f"{filepath_prefix}_frequencies.json", 'r') as f:
            freq_data = json.load(f)
            self.global_symptom_freq = Counter(freq_data['global_symptom_freq'])
            self.disease_symptom_count = freq_data['disease_symptom_count']
        
        print("All indexes loaded successfully!")
    
    def get_statistics(self):
        """Get statistics about the indexed data"""
        stats = {
            'total_diseases': len(self.disease_symptoms),
            'total_symptoms': len(self.symptom_diseases),
            'total_symptom_combinations': len(self.symptom_combinations),
            'avg_symptoms_per_disease': np.mean(list(self.disease_symptom_count.values())) if self.disease_symptom_count else 0,
            'max_symptoms_per_disease': max(self.disease_symptom_count.values()) if self.disease_symptom_count else 0,
            'min_symptoms_per_disease': min(self.disease_symptom_count.values()) if self.disease_symptom_count else 0
        }
        return stats

