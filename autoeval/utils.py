import pandas as pd


def ru_sense_with_eng_meanings_in_brackets(sset: pd.DataFrame):
    """Collect meanings descriptions from the accepted dataframe subset as a single string in the form:
    1) ru-meaning (en-value)

    For example:
    1) лаять (to bark. To produce ...)

    Args:
        sset (pd.DataFrame): subset from the original homonyms dataframe.

    Returns:
        str: Meanings written as a numbered list.
    """
    count = sset.shape[0]
    str_senses = ''
    # 1) brief ru-translation (en-value)
    for i, (ru_translation, long_sense) in enumerate(zip(sset['ru_value'].tolist(), sset['en_description'].tolist())):
        s = f"{i + 1}) {ru_translation} ({long_sense[0].lower() + long_sense[1:]})"
        if i < count - 1:  # not the last prompt? — add \n
            s += '\n'
        str_senses += s
    return str_senses


def ru_sense_with_ru_meanings_in_brackets(sset: pd.DataFrame):
    """Collect meanings descriptions from the accepted dataframe subset as a single string in the form:
    1) ru-meaning (en-value)

    For example:
    1) лаять (издавать лай...)

    Args:
        sset (pd.DataFrame): subset from the original homonyms dataframe.

    Returns:
        str: Meanings written as a numbered list.
    """
    count = sset.shape[0]
    str_senses = ''
    # 1) brief ru-translation (en-value)
    for i, (ru_translation, long_sense) in enumerate(zip(sset['ru_value'].tolist(), sset['ru_description'].tolist())):
        s = f"{i + 1}) {ru_translation} ({long_sense[0].lower() + long_sense[1:]})"
        if i < count - 1:  # not the last prompt? — add \n
            s += '\n'
        str_senses += s
    return str_senses


def only_eng_meaning(sset: pd.DataFrame):
    """Collect meanings descriptions from the accepted dataframe subset as a single string in the form:
    1) en-value

    For example:
    1) to bark. To produce ...

    Args:
        sset (pd.DataFrame): subset from the original homonyms dataframe.

    Returns:
        str: Meanings written as a numbered list.
    """
    count = sset.shape[0]
    str_senses = ''
    # 1) brief ru-translation (en-value)
    for i, long_sense in enumerate(sset['en_description'].tolist()):
        s = f"{i + 1}) {long_sense[0].lower() + long_sense[1:]}"
        if i < count - 1:  # not the last prompt? — add \n
            s += '\n'
        str_senses += s
    return str_senses
