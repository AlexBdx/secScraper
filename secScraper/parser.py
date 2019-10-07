import re
import numpy as np
import copy


class stage_2_parser():
    """
    Parser object. Acts on Stage 1 data.
    """
    
    def __init__(self, s):
        self.s = s
    
    def parse(self, parsed_report, verbose=False):
        """
        Parse the text in a report. The text of each section will be placed in a different dict key.

        :param parsed_report: the text, as a giant str
        :param verbose: Increase the amount of printing to the terminal
        :return: dict containing the parsed report with all the text by section. Metadata is in '0'
        """

        text = parsed_report['input']
        text = text.lower()
        finds = []
        if parsed_report['0']['type'] == '10-Q':
        
            # 1. Setup the giant regex to use to parse all potential sections in the report
            # 1.1. List of all possible titles
            titles = {
                '_i_1': 'financial statements',
                '_i_2': 'management s discussion and analysis of financial condition and results of operations',
                '_i_3': 'quantitative and qualitative disclosures about market risk',
                '_i_4': 'controls and procedures',
                'ii_1': 'legal proceedings',
                'ii_1a': 'risk factors',
                'ii_2': 'unregistered sales of equity securities and use of proceeds',
                'ii_3': 'defaults upon senior securities',
                'ii_4': 'mine safety disclosures',
                'ii_5': 'other information',
                'ii_6': 'exhibits'
            }
            all_sections_10q = list(titles.keys())

            # 1.2. Create the regex
            pattern = []
            for key in titles:
                # Need to parse all potential sections in case they are present.
                prefix = r'([\n\r] ?| {2,})'  # Is {3,} better?
                suffix = r'(?![a-z0-9\[\]\(\)])[\.\- ][ \n]*'
                # Because the item numbers are not unique, we NEED to include at least one word of the title
                regex = r'{}item {}{}{}'.format(prefix, key[3:], suffix, titles[key].split()[0])
                # print(regex)
                pattern.append(regex)
            pattern = r'|'.join(pattern)
            pattern = re.compile(pattern)

            # 1.3. Apply the regex, single left to right pass
            res = {section: [] for section in titles}  # Will contain all the parsed data
            for m in re.finditer(pattern, text):  # All the magic happens here!
                last_word = re.findall(r'\w+$', m.group())  # Used to backcalculate the section.

                # Sanity checks
                if len(last_word) > 1:
                    print(last_word)
                    raise ValueError('[ERROR] There should not be more than one set of numbers in the matched title.')
                elif len(last_word) == 0:
                    print(last_word)
                    raise ValueError('[ERROR] This match could not be allocated')
                
                # Find the corresponding section
                corresponding_section = 0
                for k, v in titles.items():
                    if v.split()[0] == last_word[0]:
                        corresponding_section = k
                        break
                if corresponding_section == 0:
                    print(last_word)
                    raise ValueError("[ERROR] Could not find where |{}| goes".format(last_word[0]))
                else:
                    res[corresponding_section].append(m.span())
            
            # II. Now we get serious. Purge the ToC of it exists
            # verbose=True
            # print(res)
            if verbose:
                print("[INFO] Before removing the toc:", res)
            original_res = copy.deepcopy(res)
            res = {k: v for k, v in res.items() if len(v)}
            # print(list(res.keys()))
            # Extract the Table of Content, if any
            # The gist is that if it exists, all the populated keys should follow each other in order
            # print(all_sections_10q)
            # Remove the sections that are empty so we can iterate over non-zero sections

            # Hypothesis: 1a is not optional - financial statements should not be
            # if I.1. has two entries and the second is after the 1st last entry -> toc!
            # then rm all [0] entries, then re-delete all zero entries
            # else no toc and do nothing
            full_sect = list(res.keys())
            
            # Make sure you got something. If that is not the case, might just be a completely different template.
            try:
                assert len(full_sect)
            except:
                print("[ERROR] Here is full_sect: |{}|".format(full_sect))
                print("[ERROR] Original res:", original_res)
                raise
            
            if len(res[full_sect[0]]) >= 2:
                if res[full_sect[-1]][0][1] < res[full_sect[0]][1][0]:
                    # There is a toc!
                    # print("[INFO] Found a ToC!")
                    for v in res.values():
                        del v[0]  # Remove all first sections
                    res = {k: v for k, v in res.items() if len(v)}
            else:
                # print("[INFO] No ToC found")
                pass
            
            # Extra step: make sure the first elements go in increasing order.
            try:
                res = clean_first_markers(res)
            except Exception as e:
                print('[ERROR] {} in parser.clean_first_markers (10-Q)'.format(e))
                print("This is the res\n", res)
                raise
            
            if verbose:
                print("[INFO] After removing the toc:", res)

            finds = [len(value) for value in res.values()]

            # Shrink the list of sections to review
            all_sections_10q = [k for k in all_sections_10q if k in res.keys()]
            
            # Extract the text for all the sections that we identified
            previous_start = 0
            for idx in range(len(all_sections_10q)-1):
                if all_sections_10q[idx] in self.s['sections_to_parse_10q']:  # Did we request to parse this section?
                    start = 0  # used for the data extraction
                    stop = 0
                    for span in res[all_sections_10q[idx]]:  # Go through all the titles found, in order
                        if span[1] > previous_start:  # found a starting point
                            start = span[1]
                            for span_next in res[all_sections_10q[idx+1]]:  # Same
                                if span_next[0] > start:
                                    stop = span_next[0]
                                    break  # Found a stopping point
                                else:
                                    del res[all_sections_10q[idx+1]][idx]
                            break  # Found a starting point but not nessarily a stopping point!
                            
                    if start and stop:  # 
                        assert stop > start
                        parsed_report[all_sections_10q[idx]] = text[start:stop]
                    else:
                        raise ValueError('This start {} and stop {} combination is invalid for 10-Q section {}'
                                         .format(start, stop, all_sections_10q[idx]))
                    previous_start = stop
            
            # Backward pass: if there are some sections that were expected and did not get populated
            # we populate them with a small statement.
            for section in self.s['sections_to_parse_10q']:
                try:
                    assert len(parsed_report[section]) > 0
                except KeyError:
                    # print("[WARNING] Section {} was found to be empty.".format(section))
                    parsed_report[section] = "Nothing found for this section."
                except AssertionError:
                    raise AssertionError("[ERROR] Why is that section filled with an empty text?")
                except:
                    raise
            
            # Delete the input we used and return the result
            del parsed_report['input']
            
            # DEBUG
            if 0:
                with open('/home/alex/test_10-q_{}.txt'.format(np.random.randint(1000)), 'w') as f:
                    for section in parsed_report.keys():
                        if section != '0':
                            f.write(section+'\n')
                            f.write(parsed_report[section])
                            f.write('\n================================================================\n')
                            f.write('\n==========================NEW SECTION===========================\n')
                            f.write('\n================================================================\n')
                
                    res = re.findall(regex, text)
                    finds.append(len(res))
            # print(finds)
        elif parsed_report['0']['type'] == '10-K':
            # 1. Setup the giant regex to use to parse all potential sections in the report
            # 1.1. List of all possible titles
            titles = {
                '1': 'business',
                '1a': 'risk factors',
                '1b': 'unresolved staff comments',
                '2': 'properties',
                '3': 'legal proceedings',
                '4': 'submission of matters to a vote of security holders',
                '5': 'market for registrant s common equity, related stockholder matters and issuer purchases of equity securities',
                '6': 'selected financial data',
                '7': 'management s discussion and analysis of financial condition and results of operations',
                '7a': 'quantitative and qualitative disclosures about market risk',
                '8': 'financial statements and supplementary data',
                '9': 'changes in and disagreements with accountants on accounting and financial disclosure',
                '9a': 'controls and procedures',
                '9b': 'other information',
                '10': 'directors executive officers and corporate governance',
                '11': 'executive compensation',
                '12': 'security ownership of certain beneficial owners and management and related stockholder matters',
                '13': 'certain relationships and related transactions, and director independence',
                '14': 'principal account(ant|ing) fees and services',
                '15': 'exhibits financial statement schedules'
                     }
            all_sections_10k = list(titles.keys())

            # 1.2. Create the regex
            pattern = []
            for key in titles:
                prefix = r'([\n\r] ?| {2,})'  # Is {3,} better?
                suffix = r'(?![a-z0-9\[\]\(\)])[\.\- ][ \n]*'
                regex = r'{}item {}{}{}'.format(prefix, key, suffix, titles[key].split()[0])
                pattern.append(regex)
            pattern = r'|'.join(pattern)
            pattern = re.compile(pattern)

            # 1.3. Apply the regex, single left to right pass
            res = {section: [] for section in titles}
            for m in re.finditer(pattern, text):  # All the magic happens here!
                # section_number_found = re.findall(sections_pattern_10k, m.group())
                last_word = re.findall(r'\w+$', m.group())  # Used to backcalculate the section.
                
                # Sanity checks
                if len(last_word) > 1:
                    print(last_word)
                    raise ValueError('[ERROR] There should not be more than one set of numbers in the matched title.')
                elif len(last_word) == 0:
                    print(m.group())
                    raise ValueError('[ERROR] This match: |{}| could not be allocated'.format(m.group()))
            
                # Find the corresponding section
                corresponding_section = 0
                for k, v in titles.items():
                    if v.split()[0] == last_word[0]:
                        corresponding_section = k
                        break
                if corresponding_section == 0:
                    print(last_word)
                    raise ValueError("[ERROR] Could not find where |{}| goes".format(last_word[0]))
                else:
                    res[corresponding_section].append(m.span())
            
            # II. Now we get serious. Purge the ToC of it exists
            if verbose:
                print("[INFO] Before removing the toc:", res)
            original_res = copy.deepcopy(res)
            res = {k: v for k, v in res.items() if len(v)}
            # Extract the Table of Content, if any
            # The gist is that if it exists, all the populated keys should follow each other in order
            # print(all_sections_10q)
            # Remove the sections that are empty so we can iterate over non-zero sections

            # Hypothesis: 1a is not optional - financial statements should not be
            # if I.1. has two entries and the second is after the 1st last entry -> toc!
            # then rm all [0] entries, then re-delete all zero entries
            # else no toc and do nothing
            full_sect = list(res.keys())
            # Make sure you got something. If that is not the case, might just be a completely different template.
            try:
                assert len(full_sect)
            except:
                print("[ERROR] Here is full_sect: |{}|".format(full_sect))
                print("[ERROR] Original res:", original_res)
                raise
            
            if len(res[full_sect[0]]) >= 2:
                if res[full_sect[-1]][0][1] < res[full_sect[0]][1][0]:
                    # There is a toc!
                    # print("[INFO] Found a ToC!")
                    for v in res.values():  # Iterate through all the sections
                        del v[0]  # Remove all first titles found - they are the ToC
                    res = {k: v for k, v in res.items() if len(v)}
            else:
                # print("[INFO] No ToC found")
                pass
            
            # Extra step: make sure the first elements go in increasing order.
            
            try:
                res = clean_first_markers(res)
            except Exception as e:
                print('[ERROR] {} in parser.clean_first_markers (10-K)'.format(e))
                print("This is the res\n", res)
                raise
            
            if verbose:
                print("[INFO] After removing the toc:", res)

            finds = [len(value) for value in res.values()]
            # print(finds)

            # Remove the sections that are empty so we can iterate over non-zero sections
            all_sections_10k = [k for k in all_sections_10k if k in res.keys()]
            # print(res)
            
            # Find the start & stop of each section 
            previous_start = 0
            for idx in range(len(all_sections_10k)-1):
                if all_sections_10k[idx] in self.s['sections_to_parse_10k']:  # Did we request to parse this section?
                    start = 0  # used for the data extraction
                    stop = 0
                    for span in res[all_sections_10k[idx]]:  # Go through all the titles found, in order
                        if span[1] > previous_start:  # found a starting point
                            start = span[1]
                            for span_next in res[all_sections_10k[idx+1]]:  # Same
                                if span_next[0] > start:
                                    stop = span_next[0]
                                    break  # Found a stopping point
                            break  # Found a starting point but not nessarily a stopping point!
                            
                    if start and stop:  # 
                        assert stop > start
                        parsed_report[all_sections_10k[idx]] = text[start:stop]
                    else:
                        raise ValueError('This start {} and stop {} combination is invalid for 10-K section {}'
                                         .format(start, stop, all_sections_10k[idx]))
                    previous_start = stop
            
            # Backward pass: if there are some 
            for section in self.s['sections_to_parse_10k']:
                try:
                    assert len(parsed_report[section]) > 0
                except KeyError:
                    # print("[WARNING] Section {} was found to be empty.".format(section))
                    parsed_report[section] = "Nothing found for this section."
                except AssertionError:
                    raise AssertionError("[ERROR] Why is that section filled with an empty text?")
                except:
                    raise
            
            # Delete the input we used and return the result
            del parsed_report['input']
            # print(parsed_report.keys())
            # print(parsed_report)
            
            # DEBUG
            if 0:
                with open('/home/alex/test_10-k_{}.txt'.format(np.random.randint(1000)), 'w') as f:
                    for section in parsed_report.keys():
                        if section != '0':
                            f.write(section+'\n')
                            f.write(parsed_report[section])
                            f.write('\n================================================================\n')
                            f.write('\n==========================NEW SECTION===========================\n')
                            f.write('\n================================================================\n')

        else:
            raise ValueError('[ERROR] No stage 2 parser for report type {}!'.format(parsed_report['0']['type']))
        
        if verbose:
            if len(list(set(finds))) != 1 or list(set(finds))[0] != 2:
                print("[WARNING] Issues parsing")
                # raise  # Figure it out!

        return parsed_report


def clean_first_markers(res):
    """
    In the event that a ToC was found, this will remove every first entry in the values of res. That means that all the
    location related to the titles in the ToC will be removed.

    :param res: dict, keys are sections to parse and contain the locations where the titles were found in the text.
    :return: Filtered version of res without the ToC locations
    """
    # The goal is to layer the first marker in ascending order and remove early references to them.
    sections = list(res.keys())
    # start = sections[0][1]
    for idx in range(len(sections)-1):
        counter_delete = 0
        for markers in res[sections[idx+1]]:
            if markers[0] < res[sections[idx]][0][1]:
                counter_delete += 1
            else:
                break
        for _ in range(counter_delete):
            del res[sections[idx+1]][0]
    return res

