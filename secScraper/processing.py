from datetime import datetime
from secScraper import metrics
from secScraper import parser


def process_cik(data):
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
        print("idx_first_qtr: {} | idx_last_qtr: {} | lag: {}".format(idx_first_qtr, idx_last_qtr, s['lag']))

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
    sample = 100

    for m in s['metrics']:
        # Should use a decorator here
        if m == 'diff_jaccard':
            section_result[m] = metrics.diff_jaccard(current_text, previous_text)
        elif m == 'diff_cosine_tf':
            section_result[m] = metrics.diff_cosine_tf(current_text, previous_text)
        elif m == 'diff_cosine_tf_idf':
            section_result[m] = metrics.diff_cosine_tf_idf(current_text, previous_text)
        elif m == 'diff_minEdit':
            section_result[m] = metrics.diff_minEdit(current_text[:sample], previous_text[:sample])
        elif m == 'diff_simple':
            section_result[m] = metrics.diff_simple(current_text[:sample], previous_text[:sample])
        elif m == 'sing_LoughranMcDonald':
            section_result[m] = metrics.sing_sentiment(current_text, lm_dictionary)
        else:
            raise ValueError('[ERROR] Requested method has not been implemented!')
    return section_result


def average_report_scores(result, word_count, s):
    """
    Calculate the weighted average for each requested metric.

    :param result: dictionary with the text of each section and the metadata
    :param word_count: dictionary with number of words in each parsed section
    :param s: Settings dictionary
    :return: averaged score for each of the requested metrics
    """
    final_result = {m: 0 for m in s['metrics']}
    assert final_result != {}
    assert result.keys() == word_count.keys()
    
    # Create a few totals for the weighted averages
    stc = {k: v[0] for k, v in word_count.items()}  # stp: section_total_current
    stp = {k: v[1] for k, v in word_count.items()}  # stp: section_total_previous
    sts = sum(stc.values())  # section_total_single, basically nb words in all sections of interest in current text
    _std = {k: v[0] + v[1] for k, v in word_count.items()}  # temp
    std = sum(_std.values())  # section_total_diff, basically nb words in all sections of interest in both text
    
    # Weight each metric by the number of words involved in its computation.
    for section in result.keys():
        for m in final_result.keys():
            if m[:4] == 'sing':
                try:
                    final_result[m] += result[section][m]*(stc[section]/sts)  # Consider only the nb words in current
                except:
                    print(result[section][m], (stc[section]/sts))
                    raise
            elif m[:4] == 'diff':  # Divide by the total nb or words involved in both sections
                final_result[m] += result[section][m]*((stc[section]+stp[section])/std)
            else:
                raise ValueError('[ERROR] This type of operation is not supported. How do I average it?')
    
    # Sanity check: make sure the values are meaningful
    assert final_result != {}
    for m in final_result.keys():
        if m[:4] == 'sing':  # else case already handled above
            try:
                assert -1 - s['epsilon'] <= final_result[m] <= 1 + s['epsilon']
            except:
                print(final_result)
                raise
        elif m[:4] == 'diff':
            try:
                assert -1 - s['epsilon'] <= final_result[m] <= 1 + s['epsilon']
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
    if s['differentiation_mode'] == 'monthly':  # Reports could be the same or different
        sections_to_consider = s['intersection_table']['10-K']
        result = {section: {} for section in sections_to_consider}  # 10-K notation
        
        for idx in range(len(sections_to_consider)):            
            current_section = s['intersection_table'][current['0']['type']][idx]
            previous_section = s['intersection_table'][previous['0']['type']][idx]

            try:
                current_text, previous_text = normalize_texts(current[current_section], previous[previous_section])
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
    
    elif s['differentiation_mode'] == 'yearly':
        assert current['0']['type'] == previous['0']['type']
        sections_to_consider = s['straight_table'][current['0']['type']]
        report_type = current['0']['type']
        result = {section: {} for section in s['straight_table'][report_type]}  # 10-K notation
        
        for idx in range(len(s['straight_table'][report_type])):
            current_section = s['straight_table'][report_type][idx]
            previous_section = s['straight_table'][report_type][idx]
            # print("Working on {}".format(tuple((current_section, previous_section))))
            
            # Verify that there is text allocated for each section.
            # If not, add a little something
            if current_section not in current.keys():
                # print("[WARNING] Current section {} not found".format(current_section))
                current[current_section] = "Nothing found for this section."

            if previous_section not in previous.keys():
                # print("[WARNING] Previous section {} not found".format(current_section))
                previous[previous_section] = "Nothing found for this section."
            current_text, previous_text = normalize_texts(current[current_section], previous[previous_section])

            word_count[sections_to_consider[idx]] = [len(current_text.split()), len(previous_text.split())]
            result[s['straight_table'][report_type][idx]] = calculate_metrics(current_text, previous_text, s, lm_dictionary)
    else:
        raise ValueError('[ERROR] This differentiation mode is unknown!')
    
    # Final step: we take the average of each metric
    final_result = average_report_scores(result, word_count, s)    

    # Sanity checks
    assert type(final_result) == dict
    assert len(final_result) == len(s['metrics'])
    for key in final_result.keys():
        try:
            assert -1 - s['epsilon'] <= final_result[key] <= 1 + s['epsilon']
        except:
            print("=========================\n\n\n\n\n\n\n\n\n\nFINAL RESULT", final_result)
            raise
    # Transfer the metadata
    final_result['0'] = current['0']
    print(final_result)
    
    return final_result  # Simple dictionary


def normalize_texts(current_text, previous_text):
    """
    Remove all extra white spaces, \r, \n and \t that could be left and substitute by a single whitespace.

    :param current_text: First string
    :param previous_text: Second string
    :return: tuple with both string in preserved order
    """

    return " ".join(current_text.split()), " ".join(previous_text.split())

