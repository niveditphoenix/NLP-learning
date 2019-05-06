def fun_percent_change(raw_text, mod_text):
    import numpy as np
    
    pct_change = ((len(raw_text) - len(mod_text)) / len(raw_text))*100
    return print('Percent Change is: ', np.round(pct_change, 2))


def fun_get_data(passedData):
    # Get Data
    print('<< 1. Reading Data >>')
    with open(passedData, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    print('Actual Length of Text : ', len(text))
    return text

	
def fun_clean_stopwords_punct(original_text):
    print('*'*30)
    print('<< 2. Applying StopWords and Punctuation removal >>')
    
    from nltk.corpus import stopwords
    from string import punctuation
    
    stop = stopwords.words('english')
    _stopwords = set(stop + list(punctuation))
    clean_text = ' '.join(_text for _text in original_text.split() if _text not in _stopwords)
    
    print('Length post removing stopwords and punctuation : ',len(clean_text))
    print(fun_percent_change(original_text, clean_text))
    return clean_text


def fun_clean_lemmatization(passedText, original_text):
    print('*'*30)
    print('<< 3. Applying Lemmatization >>')
    
    from nltk.stem import WordNetLemmatizer
    
    lem = WordNetLemmatizer()
    clean_text = ' '.join(lem.lemmatize(_text, pos='v') for _text in passedText.split())
    
    print('Length of text post applying Lemmatization : ', len(clean_text))
    print(fun_percent_change(original_text, clean_text))
    return clean_text


def fun_clean_removing_unwantedWords(passedText, original_text):
    print('*'*30)
    print('<< 4. Applying Removal of Unwanted Characters >>')
    
    to_remove = ['\ufeff','\n','`','~','@','#','%','^','*','--']
    clean_text_char_rem = ''
    
    for i,char in enumerate(to_remove):
        if i == 0:
            clean_text = passedText.replace(char, '')
        else:
            clean_text = clean_text.replace(char, '')
    
    print('Length of text post removal of Unwanted Characters : ', len(clean_text))
    print(fun_percent_change(original_text, clean_text))
    return clean_text


def fun_clean_removing_commonWords(passedText, original_text):
    print('*'*30)
    print('<< 5. Applying Removal of Common or Frequently Occuring Words >>')
    
    import pandas as pd
    
    common_word_freq = pd.Series(passedText.split()).value_counts()[:20] # taking only 1st 20 words
    
    clean_text = ' '.join(_char for _char in passedText.split() if _char not in common_word_freq.index)
    print('Length of text post removal of Common Words : ', len(clean_text))
    print(fun_percent_change(original_text, clean_text))
    return clean_text


def fun_clean_removing_rareWords(passedText, original_text):
    print('*'*30)
    print('<< 6. Applying Removal of Rare Occuring Words >>')
    
    import pandas as pd
    
    common_word_freq = pd.Series(passedText.split()).value_counts()[-20:] # taking only last 20 words
    
    clean_text = ' '.join(_char for _char in passedText.split() if _char not in common_word_freq.index)
    print('Length of text post removal of Rare Orccuring Words : ', len(clean_text))
    print(fun_percent_change(original_text, clean_text))
    return clean_text


def fun_clean_text():    
    # Get Data
    original_text = fun_get_data()
    
    # Clean Data
    # 1: Remove Stop Words and Puctuation
    clean_text = fun_clean_stopwords_punct(original_text)
    
    # 2: Lemmatization
    clean_text_lem_nltk = fun_clean_lemmatization(clean_text, original_text)
    
    # 3. Removing unwanted text
    clean_text_char_rem = fun_clean_removing_unwantedWords(clean_text_lem_nltk, original_text)
    
    # 4. Removing Common Words
    clean_text_post_commonWord_removal = fun_clean_removing_commonWords(clean_text_char_rem, original_text)
    
    # 5. Removing Rare Occuring Words
    clean_text_post_rareWord_removal = fun_clean_removing_rareWords(clean_text_post_commonWord_removal, original_text)
    
    return clean_text_post_rareWord_removal

if __name__ == '__main__':
	clean_text = fun_clean_text()