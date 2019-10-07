from datetime import datetime
from secScraper import metrics
from secScraper import parser
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, regexp_tokenize
from nltk.stem import WordNetLemmatizer

# Created just for use by the normalize_text function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def check_output(output, boundaries, s):
    assert boundaries[1] > boundaries[0]
    if boundaries[0] - s['epsilon'] < output < boundaries[1] + s['epsilon']:
        # Clip data to the [0; 1] interval
        output = min(output, boundaries[1])
        output = max(boundaries[0], output)
    else:
        raise ValueError('[ERROR] Metric score {} is out of bounds.'.format(output))
    return output


def process_cik(data, verbose=False):
    """
    Orchestrate the work to be done on a given CIK. There are a lot of intermediate steps, and this function will
    call a bunch of others from a variety of modules.

    :param data: Input parameters grouped in a list, just to avoid starmap calls, which I do not like.
    :return: processed data for a CIK
    """
    
    # 0. expand argument list
    cik = data[0]
    file_list = data[1]
    s = data[2]
    lm_dictionary = data[3]
    
    # 1. Parse all reports
    quarterly_submissions = {key: [] for key in s['list_qtr']}
    stg2parser = parser.stage_2_parser(s)
    file_list = sorted(file_list)
    
    for path_report in file_list:
        # print(path_report)
        split_path = path_report.split('/')
        qtr = (int(split_path[-3]), int(split_path[-2][3]))  # Ex: (2016, 3)

        if qtr in quarterly_submissions.keys():
            published = split_path[-1].split('_')[0]
            published = datetime.strptime(published, '%Y%m%d').date()
            type_report = split_path[-1].split('_')[1]
            if type_report in s['report_type']:
                with open(path_report, errors='ignore') as f:
                    text_report = f.read()
                parsed_report = dict()
                parsed_report['0'] = {'type': type_report, 'published': published, 'qtr': qtr}
                parsed_report['input'] = text_report
                # print(path_report)
                
                """Attempt to parse the report"""
                # parsed_report = stg2parser.parse(parsed_report)
                try:
                    parsed_report = stg2parser.parse(parsed_report)
                except:  # There can be a lot of error types coming from down below...
                    # If it fails, we need to skip the whole CIK as it becomes a real mess otherwise.
                    print("[WARNING] {} failed parsing".format(path_report))
                    return cik, {}, 1

                quarterly_submissions[qtr].append(parsed_report)
    
    
    #assert 0
    # Delete empty qtr - because not listed or delisted
    quarterly_submissions = {k: v for k, v in quarterly_submissions.items() if len(v) > 0}
    if len(quarterly_submissions) == 0:  # None of the reports were 10-Q or 10-K
        return cik, {}, 2
    idx_first_qtr = s['list_qtr'].index(sorted(list(quarterly_submissions.keys()))[0])
    idx_last_qtr = s['list_qtr'].index(sorted(list(quarterly_submissions.keys()))[-1])

    # Sanity checks: there should not be any issue here, but you never know
    for key in quarterly_submissions.keys():
        if len(quarterly_submissions[key]) == 0:
            print("[WARNING] No report were found for {} in the paths".format(key))
        elif len(quarterly_submissions[key]) > 1:
            print("[WARNING] {} reports were released in {}".format(len(quarterly_submissions[key]), key))
    
    # 2. Process the pair differences
    if idx_last_qtr < idx_first_qtr + s['lag']:
        # print("idx_first_qtr: {} | idx_last_qtr: {} | lag: {}".format(idx_first_qtr, idx_last_qtr, s['lag']))

        print("[WARNING] Not enough valid reports for CIK {} in this time_range. Skipping.".format(cik))
        quarterly_results = {}  # This CIK will be easy to remove later on
        return cik, {}, 3
    
    quarterly_results = {key: 0 for key in s['list_qtr'][idx_first_qtr+s['lag']:idx_last_qtr+1]}  # Include last index
    assert idx_last_qtr >= idx_first_qtr+s['lag']
    for current_idx in range(idx_first_qtr+s['lag'], idx_last_qtr+1):
        previous_idx = current_idx - s['lag']
        current_qtr = s['list_qtr'][current_idx]
        previous_qtr = s['list_qtr'][previous_idx]
        
        try:
            submissions_current_qtr = quarterly_submissions[current_qtr]
            submissions_previous_qtr = quarterly_submissions[previous_qtr]
        except:
            print("This means that for a quarter, we only had an extra document not a real 10-X")
            return cik, {}, 4
        try:
            assert len(submissions_current_qtr) == 1
            assert len(submissions_previous_qtr) == 1
        except:
            print("Damn should not have crashed here...")
            return cik, {}, 5
        if verbose:
            print("[INFO] Comparing current qtr {} to qtr {} from {} quarter ago."
              .format(s['list_qtr'][current_idx], s['list_qtr'][previous_idx], s['lag']))
        
        final_result = analyze_reports(submissions_current_qtr[0], submissions_previous_qtr[0], s, lm_dictionary)
        quarterly_results[current_qtr] = final_result
    return cik, quarterly_results, 0


def calculate_metrics(current_text, previous_text, s, lm_dictionary):
    """
    Calculate the metrics for a given pair of section text.

    :param current_text: string of text (from the current qtr parsed report's section)
    :param previous_text: string of text (from the previous qtr parsed report's section)
    :param s: Settings dictionary
    :param lm_dictionary: Sentiment analysis dictionary
    :return:
    """
    section_result = {m: 0 for m in s['metrics']}
    sample = 200  # ARTIFICIAL cap on number of words

    need_normalized_text = {'diff_jaccard', 'diff_gfg_editDistDP', 'diff_simple'}
    need_raw_text = {'diff_sk_cosine_tf', 'diff_sk_cosine_tf_idf', 'sing_LoughranMcDonald'}
    if need_normalized_text.intersection(s['metrics']):  # need_normalized_text specific stuff
        p_current_text = normalize_text(current_text, rm_stop_words=s['stop_words'], lemmatize=s['lemmatize'])
        p_previous_text = normalize_text(previous_text, rm_stop_words=s['stop_words'], lemmatize=s['lemmatize'])
    if need_raw_text.intersection(s['metrics']):  # need_raw_text specific stuff
        sw = stop_words if s['stop_words'] else None
        # No lemmatization for TF and TF-IDF
        

    for m in s['metrics']:
        # Should use a decorator here
        if m in s['diff_metrics']:
            if m == 'diff_jaccard':
                section_result[m] = metrics.diff_jaccard(p_current_text, p_previous_text)
            #elif m == 'diff_cosine_tf':
                #section_result[m] = metrics.diff_cosine_tf(current_text, previous_text)
            elif m == 'diff_sk_cosine_tf':
                section_result[m] = metrics.diff_sk_cosine_tf(current_text, previous_text, sw)
            elif m == 'diff_sk_cosine_tf_idf':
                section_result[m] = metrics.diff_sk_cosine_tf_idf(current_text, previous_text, sw)
            #elif m == 'diff_cosine_tf_idf':
                #section_result[m] = metrics.diff_cosine_tf_idf(current_text, previous_text)
            elif m == 'diff_gfg_editDistDP':
                section_result[m] = metrics.diff_gfg_editDistDP(p_current_text[:sample], p_previous_text[:sample])
                if sample < len(p_current_text) or sample < len(p_previous_text):
                    print("[WARNING] Text was cut. Current: {}/{} used | Previous: {}/{} used".format(sample, len(p_current_text), sample, len(p_previous_text)))
            #elif m == 'diff_minEdit':
                #section_result[m] = metrics.diff_minEdit(current_text[:sample], previous_text[:sample])
            elif m == 'diff_simple':
                section_result[m] = metrics.diff_simple(current_text[:sample], previous_text[:sample])
            else:
                raise ValueError('[ERROR] Requested diff method has not been implemented!')
            
            # Sanity check on the returned result.
            section_result[m] = check_output(section_result[m], (0, 1), s)
        elif m in s['sing_metrics']:
            if m == 'sing_LoughranMcDonald':
                section_result[m] = metrics.sing_sentiment(current_text, lm_dictionary)
            else:
                raise ValueError('[ERROR] Requested sing method has not been implemented!')
            section_result[m] = check_output(section_result[m], (-1, 1), s)
        else:
            raise ValueError('[ERROR] Method is not diff nor sing, what is it?')
        
        
    return section_result


def average_report_scores(result, word_count, s):
    """
    Calculate the weighted average for each requested metric.

    :param result: dictionary with the text of each section and the metadata
    :param word_count: dictionary with number of words in each parsed section
    :param s: Settings dictionary
    :return: averaged score for each of the requested metrics
    """
    #final_result = {m: 0 for m in s['metrics']}
    final_result = {section: {} for section in result.keys()}
    final_result['total'] = {m: 0 for m in s['metrics']}
    assert final_result != {}
    assert result.keys() == word_count.keys()
    
    # Create a few totals for the weighted averages
    stc = {k: v[0] for k, v in word_count.items()}  # stc: word count for each section in current
    stp = {k: v[1] for k, v in word_count.items()}  # stp: word count for each section in previous
    total_wc_current = sum(stc.values())  # section_total_single, basically nb words in all sections of interest in current text
    # section_total = {k: sum(v) for k, v in word_count.items()}  # Add word count of both documents
    total_wc_both = sum(stc.values()) + sum(stp.values())  # overall number of words in all sections
    
    # Weight each metric by the number of words involved in its computation.
    nb_sections = len(result.keys())
    # assert nb_sections == 2
    
    for section in result.keys():
        for m in s['metrics']:
            if m in s['sing_metrics']:
                # wc_averaged_metric = result[section][m]*(stc[section]/total_wc_current)
                wc_averaged_metric = result[section][m]/nb_sections
            elif m in s['diff_metrics']:  # Divide by the total nb or words involved in both sections
                # wc_averaged_metric = result[section][m]*((stc[section]+stp[section])/total_wc_both)
                wc_averaged_metric = result[section][m]/nb_sections
            else:
                raise ValueError('[ERROR] This type of operation is not supported. How do I average it?')
            final_result[section][m] = wc_averaged_metric  # Local value
            final_result['total'][m] += wc_averaged_metric  # Build a total count
    
    # Sanity checks: make sure the values are meaningful
    assert final_result != {}
    for m in final_result['total'].keys():
        if m in s['sing_metrics']:  # else case already handled above
            try:
                assert -1 - s['epsilon'] <= final_result['total'][m] <= 1 + s['epsilon']
            except:
                print(final_result)
                raise
        elif m in s['diff_metrics']:
            try:
                assert -1 - s['epsilon'] <= final_result['total'][m] <= 1 + s['epsilon']
            except:
                print("\n\n\n\n\n\n\n\n\n\nFINAL RESULT|{}|".format(final_result))
                raise
    return final_result


def analyze_reports(current, previous, s, lm_dictionary):
    """
    Calculate the difference between the two reports.

    :param current: dictionary containing the parsed current report + metadata
    :param previous: dictionary containing the parsed previous report + metadata
    :param s: Settings dictionary
    :param lm_dictionary: Sentiment analysis dictionary
    :return: dictionary containing the metadata and the score for each metric required
    """

    # We need to calculate the same things at the same time for comparison purposes. 
    word_count = dict()  # Counts the number of words in each section
    if s['differentiation_mode'] == 'quarterly':  # Reports could be the same or different    
        assert current['0']['type'] != '10-K' or previous['0']['type'] != '10-K'
        sections_current = s['common_quarterly_sections'][current['0']['type']]
        sections_previous = s['common_quarterly_sections'][previous['0']['type']]
    elif s['differentiation_mode'] == 'yearly':
        assert current['0']['type'] == previous['0']['type']
        sections_current = s['common_yearly_sections'][current['0']['type']]
        sections_previous = s['common_yearly_sections'][previous['0']['type']]
    else:
        raise ValueError('[ERROR] This differentiation mode is unknown!')
        
    sections_to_consider = zip(sections_current, sections_previous)
    
    result = {section: {} for section in sections_current}  # current report notation
    
    #for idx in range(len(sections_to_consider)):            
    for section_current, section_previous in sections_to_consider:
        if section_current not in current.keys():
            print("[WARNING] Nothing found in section {} (current) of {} published on {}"
            .format(section_current, current['0']['type'], current['0']['published']))
            current[section_current] = "Nothing found for this section."
        if section_previous not in previous.keys():
            # print("[WARNING] Previous section {} not found".format(current_section))
            print("[WARNING] Nothing found in section {} (previous) of {} published on {}"
            .format(section_previous, current['0']['type'], current['0']['published']))
            previous[section_previous] = "Nothing found for this section."
        
        current_text, previous_text = current[section_current], previous[section_previous]
        
        word_count[section_current] = [len(current_text.split()), len(previous_text.split())]
        result[section_current] = calculate_metrics(current_text, previous_text, s, lm_dictionary)
        """
        try:
            #current_text, previous_text = current[section], previous[section]
            current_text, previous_text = current[section_current], previous[section_previous]
            # current_text, previous_text = normalize_text(current[section]), normalize_text(previous[section])
        except KeyError:
            if current_section == 'ii_1a' or previous_section == 'ii_1a':
                # That means there were no update on the 10-Q
                # Not great but for now let's give it a similarity of 1
                print("Typical issue - we will fill the section_result for this 10-Q manually")
                section_result = {m: 1 for m in s['metrics']}
                result[sections_to_consider[idx]] = section_result
                continue
            else:
                raise KeyError('[ERROR] Something went wrong')
        word_count[sections_to_consider[idx]] = [len(current_text.split()), len(previous_text.split())]
        result[sections_to_consider[idx]] = calculate_metrics(current_text, previous_text, s, lm_dictionary)
        """
    
    """
    elif s['differentiation_mode'] == 'yearly':
        assert current['0']['type'] == previous['0']['type']
        report_type = current['0']['type']
        sections_to_consider = s['straight_table'][report_type]
        
        result = {section: {} for section in sections_to_consider}
        
        # for idx in range(len(sections_to_consider)):
        for section in sections_to_consider:
            # print("Working on {}".format(tuple((current_section, previous_section))))
            
            # Verify that there is text allocated for each section.
            # If not, add a little something
            if section not in current.keys():
                # print("[WARNING] Current section {} not found".format(current_section))
                print("[WARNING] Nothing found in section {} of {} published on {}"
                .format(section, current['0']['type'], current['0']['published']))
                current[section] = "Nothing found for this section."

            if section not in previous.keys():
                # print("[WARNING] Previous section {} not found".format(current_section))
                print("[WARNING] Nothing found in section {} of {} published on {}"
                .format(section, current['0']['type'], current['0']['published']))
                previous[section] = "Nothing found for this section."
            
            current_text, previous_text = current[section], previous[section]
            # current_text, previous_text = normalize_text(current[section]), normalize_text(previous[section])

            word_count[section] = [len(current_text.split()), len(previous_text.split())]
            result[section] = calculate_metrics(current_text, previous_text, s, lm_dictionary)
    """
    
    
    # RESULT AVERAGING - simple average. Word count has already be taken into account when
    # calculating the metrics...
    #final_result = average_report_scores(result, word_count, s)  # ...so this is not correct.
    total = {m: 0 for m in s['metrics']}
    nb_sections = len(result.keys())  # Number of sections for averaging
    for section in result.keys():
        for m in s['metrics']:
            total[m] += result[section][m]/nb_sections
    result['total'] = total
    #print(final_result)
    #print(result)
    #assert 0


    """ [TBR] Sanity checks are done within the function now
    # Sanity checks
    assert type(final_result) == dict
    assert len(final_result) == len(sections_current) + 1
    for key in final_result['total'].keys():
        try:
            assert -1 - s['epsilon'] <= final_result['total'][key] <= 1 + s['epsilon']
        except:
            print("=========================\n\n\n\n\n\n\n\n\n\nFINAL RESULT", final_result)
            raise
    # Transfer the metadata
    final_result['0'] = current['0']
    # print(final_result)
    
    return final_result  # Simple dictionary
    """
    return result


def normalize_text(text, tokenizer='normal', rm_stop_words=False, lemmatize=False):
    """
    Transform a string of text to a uniform format through:
    - smart tokenization (handles punctuation)
    - (optional) remove stop words
    - (optional) lemmatize

    :param text: long string of text coming from a report's section
    :param rm_stop_words: Activate stop words removal
    :param lemmatize: Activate lemmatization
    :return: list
    """
    
    # 1. Basic processing
    if tokenizer == 'normal':
        word_tokens = word_tokenize(text)
    elif tokenizer == 'regex':
        word_tokens = regexp_tokenize(text, "[\w']+")
    else:
        raise ValueError('[ERROR] Tokenizer type {} is not implemented'.format(tokenizer))
    word_tokens = [word.lower() for word in word_tokens]
    
    # 2. Advanced filtration: steps are independants.
    if rm_stop_words:
        word_tokens = [w for w in word_tokens if not w in stop_words]
    if lemmatize:
        word_tokens = [lemmatizer.lemmatize(word) for word in word_tokens]

    assert type(word_tokens) == list
    assert type(word_tokens[0]) == str
    
    return word_tokens

