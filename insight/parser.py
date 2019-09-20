import re
import numpy as np

class stage_2_parser():

    
    def __init__(self, s):
        self.s = s
    
    def parse(self, parsed_report, verbose=True):
        # Take a string as input
        # returns a dict of the text parsed
        # {0: <meta-data>}
        
        print(parsed_report)
        text = parsed_report['input']
        #parsed_report[1] = text
        #return parsed_report
        text = text.lower()
        finds = []
        if parsed_report['0']['type'] == '10-Q':
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
            sections_number_10q = list(titles.keys())
            sections_pattern_10q = re.compile('|'.join([key[3:] + '(?![a-z0-9\[\]\(\)])' for key in titles.keys()]))
            
            pattern = []
            for key in titles:
                #regex = 'item {}[\.\-\: ]*'.format(title[0])+title[1]
                prefix = r'([\n\r] ?| {2,})'  # Is {3,} better?
                suffix = r'(?![a-z0-9\[\]\(\)])[\.\- ][ \n]*'
                # Because the item numbers are not unique, we NEED to include at least one word of the title
                regex = r'{}item {}{}{}'.format(prefix, key[3:], suffix, titles[key].split()[0])
                #print(regex)
                pattern.append(regex)
            pattern = r'|'.join(pattern)
            pattern = re.compile(pattern)

            # Run a single pass left to right on data
            res = {section: [] for section in titles}
            for m in re.finditer(pattern, text):  # All the magic happens here!
                section_number_found = re.findall(sections_pattern_10q, m.group())
                #print(m.group().split()[2])
                
                # Sanity checks
                if len(section_number_found) > 1:
                    print(section_number_found)
                    raise ValueError('[ERROR] There should not be more than one set of numbers in the matched title.')
                elif len(section_number_found) == 0:
                    print(section_number_found)
                    raise ValueError('[ERROR] This match could not be allocated')
                
                # Try to allocate the value
                #print("[INFO] Pattern found: ", m.group())
                if '_i_'+section_number_found[0] in titles.keys():
                    if titles['_i_'+section_number_found[0]].split()[0] == m.group().split()[2]:
                        res['_i_'+section_number_found[0]].append(m.span())
                if 'ii_'+section_number_found[0] in titles.keys():
                    if titles['ii_'+section_number_found[0]].split()[0] == m.group().split()[2]:
                        res['ii_'+section_number_found[0]].append(m.span())
                
                    
                #print(section_number_found)
                    
            if verbose:
                print("[INFO] Before removing the toc:", res)
            # Extract the Table of Content, if any
            # The gist is that if it exists, all the populated keys should follow each other in order
            #print(sections_number_10q)
            for idx in range(len(sections_number_10q)-1):
                # Check that both are populated

                if len(res[sections_number_10q[idx]]) >= 2:
                    if len(res[sections_number_10q[idx+1]]) >= 2:
                        # You only look at the first tuple in the list as things were found in order (left to right)
                        # and the table of content is expected to be at the start of the document.
                        # If no table of content is found after going through item 1. Business, abort search for it
                        start = res[sections_number_10q[idx]][0][1]  # End of current title
                        stop = res[sections_number_10q[idx+1]][0][0]  # Start of next title
                        text_between_sections = text[start:stop]
                        # Look for words in the text between sections. Ignore numbers (could be page number)
                        #check for words
                        inter_words = re.findall('[a-z]+', text_between_sections)
                        if len(inter_words) > len(titles[sections_number_10q[idx]].split()):
                            # This section must exist and contain something outside the toc
                            if idx == 0:  # There was no toc for the "Item 1. Business case" ==> there is no toc
                                break
                        else:
                            # Found a Table of Content entry: delete it
                            print("[WARNING] Deleted entry with key {}".format(sections_number_10q[idx]))
                            del res[sections_number_10q[idx]][0]
                    else:
                        if verbose:
                            print("[WARNING] Section {} found only {} times. No ToC in this report?"
                              .format(sections_number_10q[idx+1], len(res[sections_number_10q[idx+1]])))
                else:
                    if verbose:
                        print("[WARNING] Section {} found only {} times. No ToC in this report?"
                              .format(sections_number_10q[idx], len(res[sections_number_10q[idx]])))
            if verbose:
                print("[INFO] After removing the toc:", res)
            finds = [len(value) for value in res.values()]
            #print(finds)
            #assert 0
            
            # Remove the sections that are empty
            sections_number_10q = [k for k in sections_number_10q if len(res[k])]
            #print(sections_number_10k)
            # Sanity check: can we cover the request to parse?
            sections_to_parse_10q = self.s['sections_to_parse_10q'] if len(self.s['sections_to_parse_10q']) else sections_number_10q
            #intersection = list(set(sections_number_10k) & set(sections_to_parse_10k))
            
            intersection = []
            flag_ii_1a_missing = False
            flag_ii_6_missing = False
            for ii in range(len(sections_to_parse_10q)):
                if sections_to_parse_10q[ii] in sections_number_10q:
                    intersection.append(sections_to_parse_10q[ii])
                else:  # The parser did not find a marker of that section
                    if sections_to_parse_10q[ii] == 'ii_1a':
                        flag_ii_1a_missing = True
                        # This one gets a pass as it is not mandatory
                        print("[WARNING] There was no material update since last 10-K for Part II - Item IA.")
                    elif sections_to_parse_10q[ii] == 'ii_6':
                        flag_ii_6_missing = True
                        # This one gets a pass as I do not care
                        print("[WARNING] No exhibits were found in this 10-Q but who cares?")
                    else:
                        break

            print(intersection)
            if intersection != sections_to_parse_10q:  # Not all self.s['sections_to_parse_10k'] could be found
                if flag_ii_1a_missing or flag_ii_6_missing:
                    pass
                else:
                    raise ValueError('[ERROR] Impossible to parse {} as not all of these were identified in the text.'
                                     .format(sections_to_parse_10q[ii]))
                
            
            # Find the start & stop of each section 
            previous_start = 0
            for idx in range(len(sections_number_10q)-1):
                if sections_number_10q[idx] in sections_to_parse_10q:  # Did we request to parse this section?
                    start = 0  # used for the data extraction
                    stop = 0
                    for span in res[sections_number_10q[idx]]:  # Go through all the titles found, in order
                        if span[1] > previous_start:  # found a starting point
                            start = span[1]
                            for span_next in res[sections_number_10q[idx+1]]:  # Same
                                if span_next[0] > start:
                                    stop = span_next[0]
                                    break  # Found a stopping point
                            break  # Found a starting point but not nessarily a stopping point!
                            
                    if start and stop:  # 
                        assert stop > start
                        parsed_report[sections_number_10q[idx]] = text[start:stop]
                    else:
                        raise ValueError('This start {} and stop {} combination is invalid for section {}'
                                         .format(start, stop, sections_number_10q[idx]))
                    previous_start = stop
            
            # Delete the input we used and return the result
            del parsed_report['input']
            #print(parsed_report.keys())
            #print(parsed_report)
            
            # DEBUG
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
            #print(finds)
        elif parsed_report['0']['type'] == '10-K':
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
                '10': 'directors[\,]? executive officers and corporate governance',
                '11': 'executive compensation',
                '12': 'security ownership of certain beneficial owners and management and related stockholder matters',
                '13': 'certain relationships and related transactions, and director independence',
                '14': 'principal account(ant|ing) fees and services',
                '15': 'exhibits financial statement schedules'
                     }
            #reverse_lookup = {title[1]: title[0] for title in titles}
            sections_number_10k = list(titles.keys())
            sections_pattern_10k = re.compile('|'.join([key + '(?![a-z0-9\[\]\(\)])' for key in titles.keys()]))
            
            #print(reverse_lookup)
            # Build pattern for the whole file
            pattern = []
            for key in titles:
                #regex = 'item {}[\.\-\: ]*'.format(title[0])+title[1]
                prefix = r'([\n\r] ?| {2,})'  # Is {3,} better?
                suffix = r'(?![a-z0-9\[\]\(\)])[\.\- ][ \n]*'
                regex = r'{}item {}{}{}'.format(prefix, key, suffix, titles[key].split()[0])
                #regex = r'{}item {}{}{}'.format(prefix, title[0], suffix, title[1])
                pattern.append(regex)
            pattern = r'|'.join(pattern)
            pattern = re.compile(pattern)
            
            
            # Run a single pass left to right on data
            res = {section: [] for section in titles}
            for m in re.finditer(pattern, text):  # All the magic happens here!
                section_number_found = re.findall(sections_pattern_10k, m.group())
                if len(section_number_found) > 1:
                    print(section_number_found)
                    raise ValueError('[ERROR] There should not be more than one set of numbers in the matched title.')
                elif len(section_number_found) == 0:
                    print(section_number_found)
                    raise ValueError('[ERROR] This match could not be allocated')
                else:
                    res[section_number_found[0]].append(m.span())
            
            if verbose:
                print("[INFO] Before removing the toc:", res)
            # Extract the Table of Content, if any
            # The gist is that if it exists, all the populated keys should follow each other in order
            #print(sections_number_10k)
            
            # Need to improve this section
            # - Item 1 is always found so if there is a toc for it, there is a toc for every section present in
            # the report
            # - 
            for idx in range(len(sections_number_10k)-1):
                # Check that both are populated
                """
                print("Looking at {} and {} | length are {} and {}"
                      .format(sections_number_10k[idx], sections_number_10k[idx+1],
                              len(res[sections_number_10k[idx]]), len(res[sections_number_10k[idx+1]])))
                """
                if len(res[sections_number_10k[idx]]) >= 2:
                    if len(res[sections_number_10k[idx+1]]) >= 2:
                        # You only look at the first tuple in the list as things were found in order (left to right)
                        # and the table of content is expected to be at the start of the document.
                        # If no table of content is found after going through item 1. Business, abort search for it
                        start = res[sections_number_10k[idx]][0][1]  # End of current title
                        stop = res[sections_number_10k[idx+1]][0][0]  # Start of next title
                        text_between_sections = text[start:stop]
                        # Look for words in the text between sections. Ignore numbers (could be page number)
                        #check for words
                        inter_words = re.findall('[a-z]+', text_between_sections)
                        if len(inter_words) > len(titles[sections_number_10k[idx]].split()):
                            # This section must exist and contain something outside the toc
                            if idx == 0:  # There was no toc for the "Item 1. Business case" ==> there is no toc
                                print("[WARNING] No ToC found for this report.")
                                break
                        else:
                            # Found a Table of Content entry: delete it
                            print("[WARNING] Deleted entry with key {}".format(sections_number_10k[idx]))
                            del res[sections_number_10k[idx]][0]
                    else:
                        if verbose:
                            print("[WARNING] Section {} found only {} times. No ToC in this report?"
                              .format(sections_number_10k[idx+1], len(res[sections_number_10k[idx+1]])))
                else:
                    if verbose:
                        print("[WARNING] Section {} found only {} times. No ToC in this report?"
                              .format(sections_number_10k[idx], len(res[sections_number_10k[idx]])))
            if verbose:
                print("[INFO] After removing the toc:", res)
            finds = [len(value) for value in res.values()]
            #print(finds)
            #assert 0
            
            # Remove the sections that are empty
            sections_number_10k = [k for k in sections_number_10k if len(res[k])]
            #print(sections_number_10k)
            # Sanity check: can we cover the request to parse?
            sections_to_parse_10k = self.s['sections_to_parse_10k'] if len(self.s['sections_to_parse_10k']) else sections_number_10k
            #intersection = list(set(sections_number_10k) & set(sections_to_parse_10k))
            intersection = []
            for ii in range(len(sections_to_parse_10k)):
                if sections_to_parse_10k[ii] in sections_number_10k:
                    intersection.append(sections_to_parse_10k[ii])
                else:
                    break
            
            #print(sections_to_parse_10k)
            #print(self.s['sections_to_parse_10k'])
            
            #print(intersection)
            
            #assert 0
            
            if intersection != sections_to_parse_10k:  # Not all self.s['sections_to_parse_10k'] could be found
                raise ValueError('[ERROR] Impossible to parse {} as not all of these were identified in the text.'
                                 .format(sections_to_parse_10k[ii]))
            
            # Find the start & stop of each section 
            previous_start = 0
            for idx in range(len(sections_number_10k)-1):
                if sections_number_10k[idx] in sections_to_parse_10k:  # Did we request to parse this section?
                    start = 0  # used for the data extraction
                    stop = 0
                    for span in res[sections_number_10k[idx]]:  # Go through all the titles found, in order
                        if span[1] > previous_start:  # found a starting point
                            start = span[1]
                            for span_next in res[sections_number_10k[idx+1]]:  # Same
                                if span_next[0] > start:
                                    stop = span_next[0]
                                    break  # Found a stopping point
                            break  # Found a starting point but not nessarily a stopping point!
                            
                    if start and stop:  # 
                        assert stop > start
                        parsed_report[sections_number_10k[idx]] = text[start:stop]
                    else:
                        raise ValueError('This start {} and stop {} combination is invalid for section {}'
                                         .format(start, stop, sections_number_10k[idx]))
                    previous_start = stop
            
            # Delete the input we used and return the result
            del parsed_report['input']
            #print(parsed_report.keys())
            #print(parsed_report)
            
            # DEBUG
            with open('/home/alex/test_10-k_{}.txt'.format(np.random.randint(1000)), 'w') as f:
                for section in parsed_report.keys():
                    if section != '0':
                        f.write(section+'\n')
                        f.write(parsed_report[section])
                        f.write('\n================================================================\n')
                        f.write('\n==========================NEW SECTION===========================\n')
                        f.write('\n================================================================\n')
                
            
            
            #assert 0
        else:
            raise ValueError('[ERROR] No stage 2 parser for report type {}!'.format(parsed_report['0']['type']))
        
        if verbose:
            if len(list(set(finds))) != 1 or list(set(finds))[0] != 2:
                print("[WARNING] Issues parsing")
                #raise  # Figure it out!

        return parsed_report
