"""Module for cleaning data"""

import hashlib
import argparse
import logging
from urllib.parse import urlparse
import nltk
from nltk.corpus import stopwords
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
stop_words = set(stopwords.words('spanish'))

def main(filename):
    """Main function of the receipe"""
    logger.info('Starting cleaning process')

    df = _read_data(filename)
    newspaper_uid = _extract_newspaper_uid(filename)
    df = _add_newspaper_uid_column(df, newspaper_uid)
    df = _extract_host(df)
    df = _fill_nan_data(df)
    df = _generate_uids_for_rows(df)
    df = _remove_new_lines_from_body(df)
    df = _tokenize_column(df, 'title')
    df = _tokenize_column(df, 'body')
    return df

def _extract_newspaper_uid(filename):
    """Return the newspaper uid"""

    logger.info('Extracting newspaper uid')
    newspaper_uid = filename.split('_')[0]

    logger.info('Newspaper uid detected {}'.format(newspaper_uid))
    return newspaper_uid

def _add_newspaper_uid_column(df, newspaper_uid):
    """Add the newspaper uid to a new colum"""

    logger.info('Filling the newspaper_uid colum iwth {}'.format(newspaper_uid))
    df['newspaper_uid'] = newspaper_uid
    return df

def _read_data(filename):
    """Reading the data"""

    logger.info('Reading file {}'.format(filename))

    return pd.read_csv(filename)

def _extract_host(df):
    """Extract host from the newspaper"""

    logger.info('Extracting host from urls')
    df['host'] = df['uri'].apply(lambda uri: urlparse(uri).netloc)
    return df

def _fill_nan_data(df):
    """Refilling missing data"""
    logger.info('Filling Missing titles')
    missing_titles_mask = df['title'].isna()

    missing_titles = (
        df[missing_titles_mask]['uri']
        .str.extract(r'(?P<missing_titles>[^/]+)$')
        .applymap(lambda title: title.split('-'))
        .applymap(lambda title_word_list: ' '.join(title_word_list))
    )
    df.loc[missing_titles_mask, 'title'] = missing_titles.loc[:, 'missing_titles']
    return df

def _generate_uids_for_rows(df):
    """Generate uids for each article in the dataset"""
    logger.info('Generating uids for each row')
    uids = (df
            .apply(lambda row: hashlib.md5(bytes(row['uri'].encode())), axis=1)
            .apply(lambda hash_object: hash_object.hexdigest())
   )
    df['uid'] = uids
    return df.set_index('uid')

def _remove_new_lines_from_body(df):
    """Removing new lines from the body"""
    logger.info('Removing the newline')
    stripped_body = (df
                     .apply(lambda row: row['body'], axis=1)
                     .apply(lambda body: list(body))
                     .apply(lambda letters:
                            list(map(lambda letter: letter.replace('\n', ''), letters)))
                     .apply(lambda letters: ''.join(letters))

    )
    df['body'] = stripped_body
    return df

def _tokenize_column(df, column_name):
    """Tokenize the title of each article"""
    logger.info('Creating the token title')
    tokenize = (df
                .dropna()
                .apply(lambda row: nltk.word_tokenize(row[column_name]), axis=1)
                .apply(lambda tokens:
                       list(filter(lambda token: token.isalpha(), tokens)))
                .apply(lambda tokens:
                       list(map(lambda token: token.lower(), tokens)))
                .apply(lambda word_list:
                       list(filter(lambda word: word not in stop_words, word_list)))
                .apply(lambda valid_words: len(valid_words))

                )
    df['n_tokens_{}'.format(column_name)] = tokenize
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',
                        help='The path to the dirty data',
                        type=str)
    args = parser.parse_args()
    print(main(args.filename))
