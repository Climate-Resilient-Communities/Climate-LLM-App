import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mrkdwn_analysis import MarkdownAnalyzer
import more_itertools as mit
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import pandas as pd
import jsonlines
from dotenv import load_dotenv
import cohere
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utility import read_full_markdown, num_tokens_from_string

def get_table_data(line: str) -> str:
    """Returns a complete table in string format"""
    table_pattern = re.compile(r'^\|.*\|$', re.MULTILINE)
    table_rows = table_pattern.findall(line)
    if table_rows != []:
        table = "\n".join(table_rows)
        return table
    else:
        return None

def check_next_level(level: int, line: str) -> dict:
    """Returns a dictiondary that contains levels of subheadings as key and the content of each subheading as value"""
    matches = {}
    for i in range(1, 5 - level+1):
        match_ = re.match(heading_expressions[level+i], line)
        if match_ is not None:
            matches[level + i] = match_
    return matches


def get_section_data(id: int, header_line: str, line: str, next_level: int) -> dict:
    """Returns a dictionary that contains the metadata of a section"""
    
    # Initialize empty dictionary
    metadata = {}
    
    # Log ID
    metadata['id'] = id

    # Get the correct title of the section
    if header_line == '':
      title = re.split('\n\n', line)[0]
    else:
      title = header_line
    metadata['section_title'] = title

    # Log section type
    if 'introduction' in title.lower():
        metadata['type'] = 'introduction'
    elif 'abstract' in title.lower():
        metadata['type'] = 'abstract'
    elif 'summary' in title.lower():
        metadata['type'] = 'summary'
    elif 'conclusion' in title.lower():
        metadata['type'] = 'conclusion'
    else:
        metadata['type'] = 'section'

    # Get the table if the content contains a table
    metadata['table'] = get_table_data(line)

    # Get the continuous text of the section
    metadata['content'] = line

    # Check if the content needs to be splitted into segments
    line_num_tokens = num_tokens_from_string(line, 'cl100k_base')
    if line_num_tokens <= max_token_length:
        metadata['segment'] = [line]
    else:
        # Break the content into non-overlaping segments for future retrieval 
        ## Break the content based on subheadings
        if next_level is not None :
            segments = re.split(heading_expressions[next_level], line)
            new_segments = []
            for segment in segments:
                segment_num_tokens = num_tokens_from_string(segment, 'cl100k_base')
                if segment_num_tokens <= max_token_length:
                    new_segments.extend(segment)
                else:
                    matches = check_next_level(next_level, segment)
                    if matches == {}:
                        new_segments.extend(text_splitter.split_text(segment))
                    else:
                        next_level_plus = matches.keys()[0]
                        next_segments = re.split(heading_expressions[next_level_plus], segment, flags=re.MULTILINE)
                        new_segments.extend(next_segments)
            metadata['segment'] = new_segments
        ## Break the content based on token counts 
        else:
            metadata['segment'] = text_splitter.split_text(line)
    return metadata




def heading_breakdown(file_path: str) -> list:
    """Returns a list of dictionaries that contains the metadata of each section of a MD file"""

    # Initialize MarkdownAnalyzer
    analyzer = MarkdownAnalyzer(file_path)

    # Get all the headings in the markdown file 
    headers = analyzer.identify_headers()['Header']

    # Standardize heading format
    headers_mod = [header.lower().strip(' ') for header in headers]

    # Read content from MD file
    markdown_content = read_full_markdown(file_path)

    # Initialize empty metadata_lst and id 
    metadata_lst = []
    id = 0
    for level, pattern in heading_expressions.items():

        # Breakdown content based on headings' expressions from the highest level to lower levels 
        results = re.split(pattern, markdown_content)

        # Construct metadata if there is a match with the heading expressions 
        if len(results) > 1:
            header_line = ''
            for i, line in enumerate(results):
                line_mod = line.lower().strip(' ')

                ## Check if the extracted heading is in the headings included in the markdown analysis
                if line_mod in headers_mod:
                    header_line = line
                    continue

                ## Extract heading seaprately with additional steps if the extracted heading is not included in the markdown analysis
                else:
                    id += 1
                    matches = check_next_level(level, line)
                    if matches == {}:
                        metadata_lst.append(get_section_data(id, header_line, line, None))
                        header_line = ''
                    else:
                        next_level = matches.keys()[0]
                        metadata_lst.append(get_section_data(id, header_line, line, next_level))
                        header_line = ''
        # Do nothing if there is not a match with the heading expressions 
        else:
            continue
    return metadata_lst

def identify_consecutive_grouping(int_lst: list) -> list:
    """Returns a list of lists that contains consecutive integers in int_lst"""
    return [list(group) for group in mit.consecutive_groups(int_lst)]


def recombine_small_chunks(metadata_lst: list, min_panelty: int) -> list:
    """Returns a list of dictionaries that contains the metadata of each section of a MD file"""
    # Identify segments with size less than the min_panelty 
    metadata_to_recombine = [metadata for metadata in metadata_lst if (num_tokens_from_string(metadata['content'], 'cl100k_base') <= min_panelty) and (metadata['type'] not in ['introduction', 'abstract', 'summary', 'conclusion'])]
    # Identify IDs of these segments 
    id_to_recombine = [metadata['id'] for metadata in metadata_to_recombine]
    # Determine which segments need to be recombined into one 
    consecutive_grouping = identify_consecutive_grouping(id_to_recombine)
    multiple_consecutive_grouping = [id_group for id_group in consecutive_grouping if len(id_group) > 1]
    # Recombine these groups of segments 
    for id_group in multiple_consecutive_grouping:
        if len(id_group) > 1:
            metadata_segment = []
            metadata_id = id_group
            metadata_section_title = ' '.join([metadata['section_title'] for metadata in metadata_to_recombine if metadata['id'] in id_group])
            metadata_content = ' '.join([metadata['content'] for metadata in metadata_to_recombine if metadata['id'] in id_group])
            metadata_table = ' '.join([metadata['table'] for metadata in metadata_to_recombine if metadata['id'] in id_group and metadata['table'] != None])
            metadata_segment.extend([metadata['segment'] for metadata in metadata_to_recombine if metadata['id'] in id_group])
            metadata_type = 'section'
            metadata = {
                'id': metadata_id,
                'section_title': metadata_section_title,
                'content': metadata_content,
                'table': metadata_table,
                'segment': metadata_segment,
                'type': metadata_type
            }
            metadata_lst.append(metadata)

    # Remove metadatas that are recombined from the existing metadata list 
    multiple_consecutive_grouping_flat = [id for group in multiple_consecutive_grouping for id in group]
    metadata_to_remove = [metadata for metadata in metadata_lst if metadata['id'] in multiple_consecutive_grouping_flat]
    for metadata in metadata_to_remove:
        metadata_lst.remove(metadata)
  
    return metadata_lst

def exclude_metadata(metadata_lst: list, exclusion_lst: list) -> list:
    """Return list of meatdata that contain relevant information about climate change information"""
    # Initialize the list for excluded metadata
    exclusion_metadata_lst = []
    # Extract metadata to be excluded
    for metadata in metadata_lst:
        for exclusion_str in exclusion_lst:
           if exclusion_str in metadata['section_title'].lower():
                exclusion_metadata_lst.append(metadata)
    # Remove excluded metadata from existing metadata list
    clean_metadata_lst = [metadata for metadata in metadata_lst if metadata not in exclusion_metadata_lst]
    return clean_metadata_lst

def combine_summary(clean_metadata_lst: list) -> tuple:
    """Return a tuple of combined introduction and conclusion and summaries"""
    intro_conclusion_lst = ['introduction', 'conclusion']
    summary_lst = ['abstract', 'summary']
    # Extract introduction and conclusion passages and combine into one string 
    intro_conclusion = '\n\n'.join([metadata['content'] for metadata in clean_metadata_lst if metadata['type'] in intro_conclusion_lst])
    # Extract abstract and summary passages and combine into one string
    summary = '\n\n'.join([metadata['content'] for metadata in clean_metadata_lst if metadata['type'] in summary_lst])
    return (intro_conclusion, summary)

def get_url(file_path: str, query: str) -> list:
    """Return a list of urls that are links to the original document"""
    url = []
    try:
      for url_ in search(query, tld="co.in", num=1, stop=1, pause=2):
          url.append(url_)
    except Exception as e:
        print(e)
        url = [file_path]
    return url

def map_title(file_title_dict: dict, MD_file_path:str) -> str:
    file_name = MD_file_path.rpartition('/')[-1].replace('md', 'pdf')
    title = file_title_dict[file_name] 
    return title

def gen_doc(MD_file_path: str, min_panelty: int, exclusion_lst: list, file_title_dict: dict) -> dict:
    """Return a dictionary of information about the MD file"""
    metadata_lst = heading_breakdown(MD_file_path, min_panelty)
    clean_metadata_lst = exclude_metadata(metadata_lst, exclusion_lst)
    recombined_metadata_lst = recombine_small_chunks(clean_metadata_lst, min_panelty)
    intro_conclusion, summary = combine_summary(clean_metadata_lst)
    title = map_title(file_title_dict, MD_file_path)
    query = title +  intro_conclusion + summary
    url = get_url(MD_file_path, query)
    doc = {
        'title': title,
        'file_path': MD_file_path,
        'intro_conclusion': intro_conclusion,
        'summary': summary,
        'url': url,
        'metadata': recombined_metadata_lst
    }
    return doc

def gen_docs(MD_dir: str, min_panelty: int, exclusion_lst: list, file_title_map_path: str) -> list:
    """Return a list of dictionaries of information about the MD files"""

    file_title_pd = pd.read_csv(file_title_map_path)
    file_title_dict = file_title_pd[['file_name', 'article_name']].drop_duplicates()
    file_title_dict = file_title_dict.set_index('file_name').to_dict()['article_name']

    # Initialize lists
    docs = []
    MDs_file_path = []
    # Find all the MD files in the MD_dir
    for root, dirs, files in os.walk(MD_dir):
        MD_files = [os.path.join(root, f) for f in files if f.endswith('.md')]
        MDs_file_path.extend(MD_files)
    for MD in MDs_file_path:
        doc = gen_doc(MD, min_panelty, exclusion_lst, file_title_dict)
        docs.append(doc)
    return docs

def cohere_summarization_single_try(instruction, model = "command-r-plus", temperature = 0):
    """Return a summary of a MD file"""
    response = cohere_client.chat(
            message = instruction,
            model = "command-r-plus",
            temperature = 0
        )
    return response.text

def cohere_summarization(instruction, model = "command-r-plus", temperature = 0, num_retries = 5, waiting_time = 1):
    """Return a summary of a MD file"""
    summary = ''
    for _ in range(num_retries):
        try:
            summary =  cohere_summarization_single_try(instruction, model, temperature)
            break
        except Exception as e:
            print(e)
            time.sleep(waiting_time)
    return summary

def summarize_docs(docs):
    """Return a list of dictionaries of information about each MD file with summmary metadata populated"""
    docs_lst = []
    for doc in docs:
        intro_conclusion = doc['intro_conclusion']
        summary = doc['summary']
        file_path = doc['file_path']
        content = read_full_markdown(file_path)
        if summary == '' and intro_conclusion == '':
            summary = cohere_summarization(summary_instruction.format(text = content))
            doc['summary'] = summary
        docs_lst.append(doc)
    return docs_lst

def keyword_extraction(model, text, exclusion_lst, top_n):
    """Return a list of keywords of a chunk of text"""
    try:
        keywords = model.extract_keywords(text, vectorizer=KeyphraseCountVectorizer(), keyphrase_ngram_range=(1,3),  stop_words='english',
                                          use_maxsum=True, nr_candidates=20, top_n= top_n)
        keywords = [keyword[0] for keyword in keywords if keyword[0] not in exclusion_lst and '$' not in keyword[0]]
        return keywords
    except Exception as e:
        print(e)
        return []

def keyword_extraction_iter(model, docs, exclusion_lst):
    """Return a list of keywords of introduction, conclusion, summary and each section of the MD file"""
    docs_new = []
    for doc in docs:
      metadata = doc['metadata']
      intro_conclusion = doc['intro_conclusion']
      summary = doc['summary']
      overall_keywords = keyword_extraction(model, intro_conclusion + summary, exclusion_lst, 10)
      doc['keywords'] = overall_keywords
      metadata_new = []
      for metadata_ in metadata:
        metadata_keywords = keyword_extraction(model, metadata_['content'], exclusion_lst, 5)
        metadata_['keywords'] = metadata_keywords
        metadata_new.append(metadata_)
      doc['metadata'] = metadata_new
      docs_new.append(doc)
    return docs_new

if __name__ == "__main__":

    load_dotenv("YOUR SECRET ENVIRONMENT HERE, e.g. '../../secrets.env'")
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    cohere_client = cohere.Client(COHERE_API_KEY)

    # Initialize heading expressions  
    heading_expressions = {
     1: r'\n#(\s.+)',
     2: r'\n##(\s.+)',
     3: r'\n###(\s.+)',
     4: r'\n####(\s.+)',
     5: r'\n#####(\s.+)',
     }

    max_length = 40960
    max_token_length = 1024
    min_panelty = 250


    ## Split text into manageable chunks for better indexing
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1024, chunk_overlap=0
    )

    header_exclusion_lst = ['acknowledg', 'image', 'table of contents', 'references', 'author agreement statement', 'declaration of competing interest']

    MD_dir = sys.argv[1]
    file_title_map_path = sys.argv[2]
    output_jsonl_path = sys.argv[3]

    # Preprocess MD files
    docs = gen_docs(MD_dir, min_panelty, header_exclusion_lst, file_title_map_path)

    # Summarize MD files
    summary_instruction = """
        ## Instructions
    Below there is climate change report or academia paper in Canada. Please summarize the salient points of the text and do so in a flowing high natural language quality text. Use bullet points where appropriate.

    Paraphrase the content into re-written, easily digestible sentences. Do not extract full sentences from the input text.

    The output summary should be at least 50 words and no more than 150 words

    ## Climate Change Report or Academia Paper in Canada
    {text}
    """
    summarized_docs = summarize_docs(docs)

    # Initialize the KeyBERT model
    tokenizer = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-f")
    climatebert = AutoModelForMaskedLM.from_pretrained("climatebert/distilroberta-base-climate-f")
    kw_model = KeyBERT(model=climatebert)
    kw_exclusion_lst = ['climate', 'climate change', 'global warming', 'canadian']

    # Extract Keywords
    docs_kw = keyword_extraction_iter(kw_model, summarized_docs, kw_exclusion_lst)

    with jsonlines.open(output_jsonl_path, mode='w') as writer:
        writer.write_all(docs_kw)
