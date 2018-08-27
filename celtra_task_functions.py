import numpy as np
import pandas as pd
from scipy.stats import norm
from urllib.request import urlopen
from gzip import decompress


def import_file(url):
    """
    Import, decompress and save a file from url to data frame.

    Args:
        url: Web address of data source.

    Returns:
        df: DataFrame
    """
    response = urlopen(url)

    json_file = decompress(response.read())
    df = pd.read_json(json_file, lines=True)
    return df


def preprocess_df(df):
    """
    Optimize data types and filter live purpose.

    Args:
        df: DataFrame

    Returns:
        df: DataFrame
    """
    live_sessions = df.loc[df['purpose'] == 'live', 'sessionId'].unique()
    df = df[df['sessionId'].isin(live_sessions)].drop(columns=['purpose'])

    # change hashes to unique integers
    df['sessionId'] = df['sessionId'].factorize()[0]
    df['sessionId'] = pd.to_numeric(df['sessionId'], downcast='unsigned')

    # change from object to category
    df['name'] = df['name'].astype('category')

    # convert object to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # convert object to datetime
    df['clientTimestamp'] = pd.to_datetime(df['clientTimestamp'], unit='s', errors='coerce')

    # downcast and change confusing name
    df['index'] = pd.to_numeric(df['index'], downcast='float', errors='coerce')

    #
    df['objectClazz'] = df['objectClazz'].astype('category')

    #
    df['sdk'] = df['sdk'].astype('category')

    df.reset_index(drop=True, inplace=True)
    return df


def calc_passed_time(df, start, stop):
    """
    Calculate time passed between screenShown event and the first interaction for different sessions.

    Args:
        df: pandas DataFrame
        start: timestamp for the starting boundary
        stop: timestamp for the ending boundary

    Returns:
        t_diff: Serie of time differences in seconds
    """
    df = df.loc[df['clientTimestamp'].between(start, stop),
                ['sessionId', 'clientTimestamp', 'name']].sort_values('clientTimestamp')

    t_screenShown = df.loc[df['name'] == 'screenShown',
                           ['sessionId', 'clientTimestamp']].drop_duplicates(
        'sessionId').set_index(
        'sessionId')

    t_firstInteraction = df.loc[
        df['name'] == 'firstInteraction',
        ['sessionId', 'clientTimestamp']].drop_duplicates('sessionId').set_index('sessionId')

    t_diff = (t_firstInteraction - t_screenShown).dropna()['clientTimestamp'].dt.total_seconds()
    return t_diff


def engagement_df(df):
    """
    Returns DataFrame with session timestamps and column of engagement booleans.

    Args:
        df: pandas DataFrame

    Returns:
        df: DataFrame with timestamps and engagement values
    """
    is_engaged = df.loc[(df['name'] == 'interaction') | (df['name'] == 'firstInteraction'), 'sessionId'].unique()
    df = df.loc[(df['name'] == 'adRequested'), ['sessionId', 'timestamp']]
    df['engaged'] = df['sessionId'].isin(is_engaged).drop(columns=['sessionId'])
    df.drop(columns=['sessionId'], inplace=True)
    return df


def twoSampZ(X1, sd1, n1, X2, sd2, n2):
    """
    Calculate two sided z-test for means, standard deviations and sizes of two samples.

    Args:
        X1, X2: means of samples
        std1, std2: standard deviations of samples
        n1,n2: sample sizes

    Returns:
        pval: p-value for two sided z-test
    """
    pooledSE = np.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2)
    z = (X1 - X2) / pooledSE
    pval = 2 * norm.sf(np.abs(z))
    return pval


def er_parameters(engagement_serie):
    """
    Calculate ER and standard deviation for one trial.

    Args:
        engagement_serie: pandas Serie of boolean engagement rate data

    Returns:
        er: engagement rate
        std_er: standard deviation for 1 one session with this er
        seen: number of trials (sample size)
    """
    engaged = engagement_serie.sum()
    seen = len(engagement_serie)

    er = engaged / seen
    std_er = np.sqrt(er * (1 - er))

    return er, std_er, seen


def er_stats(df, border_time):
    """
    Calculate ER difference and p-value of a two sided z-test for samples before and after the border time.

    Args:
        df: DataFrame
        border_time: timestamp of a time which seperates the data

    Returns:
        er_diff: er(after) - er(before)
        p_value: p-value for two sided z-test
        k1,k2: coefficients which have to be more than 5 for p-value to be reliable
    """
    before = df['timestamp'] < border_time

    (er1, std1, n1) = er_parameters(df.loc[before, 'engaged'])
    (er2, std2, n2) = er_parameters(df.loc[~before, 'engaged'])

    p_value = twoSampZ(er1, std1, n1, er2, std2, n2)
    er_diff = er2 - er1
    k1 = min(n1 * er1, n1 * (1 - er1))
    k2 = min(n2 * er2, n2 * (1 - er2))

    return er_diff, p_value, k1, k2


#
def attribute_df(df, attribute):
    """
    Create DataFrame with (sessionId,attribute) unique pairs and timestamps from adRequest events.
    Column 'engaged' describes engagement.

    Args:
        df: DataFrame
        attribute: attribute to group by

    Returns:
        df_merged: DataFrame ith (sessionId,attribute) unique pairs and timestamps from adRequest events.
        Column 'engaged' describes engagement.
    """

    # create dataframe without duplicates and keep only sessionId and attribute column
    df_attr = df.dropna(subset=[attribute]).drop_duplicates(
        ['sessionId', attribute])[['sessionId', attribute]]

    # get timestamps for adRequested event in specific session
    ad_timestamps = df.loc[df['name'] == 'adRequested', ['timestamp', 'sessionId']]

    # merge dataframes on sessionId
    df_merged = df_attr.merge(ad_timestamps, how='left', on='sessionId')

    # check if the user has engaged
    is_engaged = df.loc[(df['name'] == 'interaction') | (df['name'] == 'firstInteraction'), 'sessionId'].unique()
    df_merged['engaged'] = df_merged['sessionId'].isin(is_engaged)
    return df_merged


def attribute_stats(df, attribute, border_time):
    """
    Calculate z-test statistics on different groups of attributes with the selected border time.

    Args:
        df: DataFrame
        attribute: attribute to group by
        border_time: timestamp of a time which seperates the data

    Returns:
        df_stats: DataFranme consisting of columns:
            diff: er(after) - er(before)
            p_value: p-value for two sided z-test
            k1,k2: coefficients which have to be more than 5 for p-value to be reliable
    """
    df_stats = df.groupby(attribute)[['timestamp', 'engaged']].apply(er_stats, border_time).apply(pd.Series)

    df_stats = df_stats.rename(index=str, columns={0: "diff", 1: "p_value", 2: "k1", 3: "k2"}).sort_values('p_value')

    return df_stats


def download_file(url, filepath):
    """
    Combination of the other functions - look at their documentations.

    Args:
        url: Web address of data source.
        filepath: Path for saving a CSV file.

    Returns:
        df_both: DataFrame
    """
    print('Downloading and processing the file...')
    df = import_file(url)
    df = preprocess_df(df)

    sdks = attribute_df(df, 'sdk')
    objects = attribute_df(df, 'objectClazz')

    df_both = pd.merge(objects[['sessionId', 'timestamp', 'objectClazz']],
                       sdks[['sessionId', 'sdk', 'engaged']], on='sessionId')

    df_both.to_csv(filepath, index=False)
    return df_both


def read_file(filepath, sdk=None, object=None):
    """
    Load CSV file which consists of sessionIds, timestamps and attributes (sdk and objectClazz) into DataFrame
    and filter it based on input arguments.

    Args:
        filepath: path to a file
        sdk: filter DataFrame on sdk column which should match the argument
        object: filter DataFrame on objectClazz column which should match the argument

    Returns:
        df: filtered DataFrame
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if (sdk != None) & (object != None):
        df = df[(df['objectClazz'] == object) & (df['objectClazz'] == object)]
    elif sdk != None:
        df = df[df['sdk'] == sdk]
    elif object != None:
        df = df[df['objectClazz'] == object]
    else:
        pass

    return df


def get_attribute_labels(df):
    """
    Get unique attribute labels for sdk and objectClazz column.

    Args:
        df: DataFrame

    Returns:
        sdk_labels: unique labels for sdk column
        object_labels: unique labels for objectClazz column
    """
    unique_object = df['objectClazz'].unique()
    unique_sdk = df['sdk'].unique()

    sdk_labels = [{'label': x, 'value': x} for x in unique_sdk]
    object_labels = [{'label': x, 'value': x} for x in unique_object]
    return sdk_labels, object_labels


def resample_calculate_er(df, dt):
    """
    Resample data with the frequency dt and apply the function er_parameters.

    Args:
        df: DataFrame
        dt: sampling frequency in minutes

    Returns:
        df_output: DataFrame consisting of: er, standard deviation of er, sample size

    """
    df_output = pd.DataFrame()
    df_output[['er', 'std_er', 'n']] = df.set_index('timestamp').resample(f'{dt}T')['engaged'].apply(
        er_parameters).apply(pd.Series)
    return df_output


def get_axis_data(df, dt, smoothing_alpha):
    """
    Get axis data.

    Args:
        df: DataFrame
        dt: sampling frequency in minutes
        smoothing_alpha: smoothing factor alpha for the first order exponential smoothing

    Returns:
        x: timestamps which are in the middle of sampling intervals
        y: engagement rates in percentages
        y_error: standard deviation of engagement rates in percentages
        y_smoothed: exponentially smoothed engagement rates

    """
    x = pd.to_datetime(df.index) + pd.Timedelta(minutes=dt / 2)
    y = (100 * df['er'])
    y_error = 100 * (df['std_er'] / np.sqrt(df['n']))
    y_smoothed = (100 * df['er']).ewm(alpha=smoothing_alpha).mean()
    return x, y, y_error, y_smoothed
