import re
import pandas as pd

class SetswanaLemmatizer:
    def __init__(self):
        # Initialize lookup tables for exceptions
        self.passive_exceptions = {'ungwa', 'wa', 'swa', 'nwa', 'hwa'}
        self.causative_exceptions = {'tataisa', 'gaisa', 'laisa', 'fisa'}
        self.applicative_exceptions = {'bela', 'sela', 'tlhatlhela'}
        self.reciprocal_exceptions = {'pana', 'gana', 'fapaana', 'rulagana'}
        self.neuter_passive_exceptions = {'sega', 'bega', 'anega', 'pega'}
        
        # Define transformation rules
        self.transformations = [
            self._remove_plural,
            self._remove_perfect_tense,
            self._remove_passive,
            self._remove_reciprocal,
            self._remove_applicative,
            self._remove_neuter_passive,
            self._remove_causative,
            self._remove_reversal,
            self._remove_reflexive,
            self._remove_object_markers,
            self._remove_iterative,
            self._fix_mood
        ]
        
        # Define perfect tense conversions
        self.perfect_conversions = {
            'etswe': 'lwa', 'otswe': 'lwa', 'utswe': 'lwa',
            'ditse': 'tsa', 'tswitse': 'tsa',
            'elle': 'aa',
            'ntse': 'nya',
            'tshtswe': 'tshwa',
            'sitswe': 'shwa', 'sitswe': 'swa',
            'tshtse': 'tsha',
            'ntswe': 'mapwa',
            'sitse': 'sa',
            'dile': 'la', 'tse': 'la',
            'lwe': 'wa',
            'ditswe': 'tswa', 'tswitswe': 'tswa', 'tsitswe': 'tswa',
            'ile': 'a',
            'nne': 'na',
            'nwe': 'nwa'
        }
        
        # Define reflexive transformations
        self.reflexive_transforms = {
            'a': 'ika',
            'e': 'ike',
            'i': 'iki',
            'o': 'iko',
            'u': 'iku',
            'w': 'ikw',
            'g': 'ikg',
            'b': 'ip',
            'l': 'it',
            'r': 'ith',
            's': 'itsh',
            'd': 'it',
            'h': 'iph',  # simplified - paper mentions more complex cases
            'f': 'iph'
        }
    
    def lemmatize(self, word):
        """
        Lemmatize a Setswana verb by applying transformation rules in sequence
        """
        original_word = word
        changed = True
        
        # Apply transformations until no more changes occur
        while changed:
            changed = False
            for transform in self.transformations:
                new_word = transform(word)
                if new_word != word:
                    word = new_word
                    changed = True
                    break  # restart transformations after each change
        
        return word if word != original_word else original_word
    
    def _remove_plural(self, word):
        """Remove plural suffix -ng"""
        if word.endswith('ng'):
            return word[:-2]
        return word
    
    def _remove_perfect_tense(self, word):
        """Remove perfect tense suffixes"""
        if word in self.passive_exceptions:
            return word
            
        for suffix, replacement in self.perfect_conversions.items():
            if word.endswith(suffix):
                return word[:-len(suffix)] + replacement
        
        # Special case for -ile (most common perfect tense)
        if word.endswith('ile'):
            return word[:-3] + 'a'
        
        return word
    
    def _remove_passive(self, word):
        """Remove passive suffixes"""
        if word in self.passive_exceptions:
            return word
            
        # Table I transformations from the paper
        passive_transforms = {
            'biwa': 'ba', 'jwa': 'ba',
            'fiwa': 'fa', 'swa': 'fa',
            'giwa': 'ga', 'gwa': 'ga',
            'piwa': 'pa', 'tswa': 'pa',
            'miwa': 'ma', 'ngwa': 'ma',
            'niwa': 'na', 'nwa': 'na',
            'nyiwa': 'nya', 'nywa': 'nya',
            'diwa': 'tsa', 'tswa': 'tsa',
            'tliwa': 'tlha', 'tlhwa': 'tlha',
            'tliwa': 'tla', 'tlhwa': 'tla',
            'tiwa': 'ta', 'twa': 'ta',
            'siwa': 'sa', 'swa': 'sa',
            'wiwa': 'wa',
            'wa': 'a'
        }
        
        for suffix, replacement in passive_transforms.items():
            if word.endswith(suffix):
                return word[:-len(suffix)] + replacement
        
        return word
    
    def _remove_causative(self, word):
        """Remove causative suffix -is-"""
        if word in self.causative_exceptions:
            return word
            
        if word.endswith('isha'):
            return word[:-4] + 'a'
        elif word.endswith('isa'):
            return word[:-3] + 'a'
        elif word.endswith('isisa'):  # intensity form
            return word[:-5] + 'a'
            
        return word
    
    def _remove_applicative(self, word):
        """Remove applicative suffix -el-"""
        if word in self.applicative_exceptions:
            return word
            
        if word.endswith('ela'):
            return word[:-3] + 'a'
        elif word.endswith('ele'):
            return word[:-3] + 'a'
        elif word.endswith('elwa'):
            return word[:-4] + 'a'
            
        return word
    
    def _remove_reciprocal(self, word):
        """Remove reciprocal suffix -an-"""
        if word in self.reciprocal_exceptions:
            return word
            
        if word.endswith('ana'):
            return word[:-3] + 'a'
        elif word.endswith('anya'):
            return word[:-4] + 'a'
            
        return word
    
    def _remove_neuter_passive(self, word):
        """Remove neuter-passive suffixes (-eg-, -al-, -agal-, -eseg-)"""
        if word in self.neuter_passive_exceptions:
            return word
            
        if word.endswith('ega'):
            return word[:-3] + 'a'
        elif word.endswith('ala'):
            return word[:-3] + 'a'
        elif word.endswith('agala'):
            return word[:-5] + 'a'
        elif word.endswith('esega'):
            return word[:-5] + 'a'
            
        return word
    
    def _remove_reversal(self, word):
        """Remove reversal suffix -olol-"""
        # As noted in the paper, most words with -olol- are basic forms
        # So we only handle specific cases that we know need transformation
        reversal_examples = {
            'bofolola': 'bofa',
            'kopolola': 'kopa'
        }
        
        return reversal_examples.get(word, word)
    
    def _remove_iterative(self, word):
        """Remove iterative suffix -ka-"""
        if 'kaka' in word:
            return re.sub(r'kaka$', '', word)
        elif word.endswith('ka'):
            return word[:-2]
        return word
    
    def _remove_reflexive(self, word):
        """Remove reflexive prefix i- with transformations"""
        if not word.startswith('i'):
            return word
            
        # Handle reflexive transformations from Table II
        for initial, prefix in self.reflexive_transforms.items():
            if word.startswith(prefix):
                # Remove the reflexive prefix and restore the original initial
                return initial + word[len(prefix):]
        
        # Simple case: just remove 'i' prefix
        if word.startswith('i'):
            return word[1:]
        
        return word
    
    def _remove_object_markers(self, word):
        """Remove object markers (first-person n-, third-person mo-)"""
        # First-person object marker n- becomes m- before certain consonants
        if word.startswith('m') and len(word) > 1:
            next_char = word[1]
            if next_char in {'p', 'b', 'ph', 'f'}:
                return 'b' + word[2:]  # n- becomes m- and original consonant was b/p/ph/f
        
        # Third-person object marker mo- contracted to m- and b- becomes -m
        if word.startswith('mm') and len(word) > 2:
            return 'b' + word[2:]  # e.g., mmetsa -> beta
        
        if word.startswith('n'):
            return word[1:]
            
        if word.startswith('mo'):
            return word[2:]
            
        return word
    
    def _fix_mood(self, word):
        """Fix mood by replacing -e with -a"""
        if word.endswith('e'):
            return word[:-1] + 'a'
        return word


def process_transcription(transcription, lemmatizer):
    """
    Process a transcription string with numbered sentences
    Returns a new string with lemmatized words
    """
    # Split into numbered sentences
    sentences = [s.strip() for s in transcription.split('\n') if s.strip()]
    
    processed_sentences = []
    for sentence in sentences:
        # Split into number and text (e.g., "1. e kitshedimosetso...")
        parts = sentence.split('.', 1)
        if len(parts) == 2:
            num_part, text_part = parts
            # Lemmatize each word in the text
            words = text_part.split()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            # Reconstruct the sentence
            processed_sentence = f"{num_part}. {' '.join(lemmatized_words)}"
            processed_sentences.append(processed_sentence)
        else:
            # If no number, just process the text
            words = sentence.split()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            processed_sentences.append(' '.join(lemmatized_words))
    
    return '\n'.join(processed_sentences)


def process_csv(input_file, output_file):
    """
    Process the CSV file, lemmatizing the transcription column
    """
    # Read the CSV file
    df = pd.read_excel(input_file)
    
    # Initialize lemmatizer
    lemmatizer = SetswanaLemmatizer()
    
    # Process each transcription
    df['Lemmatized_Transcription'] = df['Transcription'].apply(
        lambda x: process_transcription(x, lemmatizer) if pd.notnull(x) else ''
    )
    
    # Save to new file
    df.to_excel(output_file, index=False)
    print(f"Processed data saved to {output_file}")


if __name__ == "__main__":
    # Example usage with the podcast transcriptions file
    input_filename = "podcast_transcriptions_chunked_sorted_numbered+english.xlsx"
    output_filename = "lemmatized_transcriptions.xlsx"
    
    process_csv(input_filename, output_filename)
    
    print("\nTest Cases:")
    lemmatizer = SetswanaLemmatizer()
    test_words = [
        'supiwa',    # passive of supa (point)
        'supisa',    # causative of supa
        'supisisa',  # intensity of supa
        'supela',    # applicative of supa
        'supana',    # reciprocal of supa
        'ikopa',     # reflexive of kopa (ask)
        'iphenya',   # reflexive of fenya (win)
        'robakaka',  # iterative of roba (break)
        'bofolola',  # reversal of bofa (tie)
        'rapelang',  # plural of rapela (pray)
        'itshupile', # perfect reflexive of supa
        'mmetsa',    # third-person object marker of beta (ask)
        'mpona',     # first-person object marker of bona (see)
        'palame'     # mood form of palama (climb)
    ]
    
    print("Setswana Verb Lemmatizer Test Cases")
    print("=" * 40)
    for word in test_words:
        lemma = lemmatizer.lemmatize(word)
        print(f"{word:15} → {lemma}")
