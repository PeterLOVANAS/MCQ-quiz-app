# FASTAPI API for adding, searching, updating, and deleting questions in a database.
from fastapi import FastAPI, HTTPException, Query, Body, status
from fastapi.responses import JSONResponse

# PyMilvus for database operations
from pymilvus import MilvusClient
from pymilvus import model
from pymilvus.model.reranker import JinaRerankFunction

# Custom modules
from agents import Agent

# Other libraries
from typing import Optional, List
import numpy as np
import random
from dotenv import load_dotenv # for loading environment variables
import os
import logging # USE logging
import yaml
load_dotenv()

def load_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# May add to config yaml file?
DB_NAME = config["db"]["name"]  # with .db extension
COL_NAME = config["db"]["collection"] # collection name
DESCRIPTION = config["db"]["description"]
TEMPLATE_FILE = config["files"]["template"] # Prompt template for question generation
INIT_TOP_FILE = config["files"]["default_topics"] # File with default topics
SIM_THRESHOLD = config["thresholds"]["similarity"] # For Diversity
BATCH_SIZE = config["batch"]["size"]
DIM = config["db"]["dimension"] # Embedding dimension of vectors inside a DB
ALPHA= config["thresholds"]["alpha"] # For determine the limit of questions to be reranked

# Initialize Logger_aq
logging.basicConfig(level=logging.INFO)
logger_db = logging.getLogger("Database")
logger_aq = logging.getLogger("QuestionsAPI")
logger_s = logging.getLogger("SearchAPI")

# Defining Embedding Model
openai_model = model.dense.OpenAIEmbeddingFunction(
  model_name = 'text-embedding-3-large',
  api_key = os.getenv('OPENAI_API_KEY'), 
  dimensions = 3072
)

# Defining Reranker
jina_rf = JinaRerankFunction(
    model_name="jina-reranker-v2-base-multilingual", 
    api_key=os.getenv("JINA_API_KEY")
)

# Defining Milvus Client DB
db = MilvusClient(DB_NAME)

# Selected Collection name
if not db.has_collection(COL_NAME):
    from pymilvus import CollectionSchema, FieldSchema, DataType

    schema = db.create_schema(auto_id=True)

    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name='embedding', datatype=DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field(field_name='level', datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name='question', datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name='code', datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name='opt_1', datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name='opt_2', datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name='opt_3', datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name='opt_4', datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name='correct_opt', datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name='explanation', datatype=DataType.VARCHAR, max_length=10000)
    
    index_params = db.prepare_index_params()
    index_params.add_index(field_name= 'embedding', metric_type='COSINE')
    logger_db.info(f"Creating collection {COL_NAME}")
    db.create_collection(collection_name=COL_NAME, schema=schema, dimension=DIM, index_params=index_params)
else:
    logger_db.info(f"Collection {COL_NAME} already exists.")
    db.load_collection(COL_NAME) # ***Should later release_collection()***


# Finding index that is too similar to the new embeddings
def find_idx_similarity(client, collection_name ,embeddings):
  idx_to_remove = set()

  for i, emb_vec in enumerate(embeddings):

      if i in idx_to_remove:
         continue

      search_results = client.search(
          collection_name= collection_name, 
          data = [emb_vec.tolist()],
          search_params= {'metric_type': 'COSINE'}
        )
      
      for hit in search_results[0]:
          if hit['distance'] >= SIM_THRESHOLD and hit['id'] != i:
              idx_to_remove.add(i)
  
  return idx_to_remove


# Suppose we have {'Pandas': 94.5, 'Numpy': 62.370000000000005, 'Matplotlib': 32.13}, we may get remainders like this:
# Remainder = {'Pandas': 0.5, 'Numpy': 0.37000000000000455, 'Matplotlib': 0.13000000000000256}
# Distribute based on these current remainders. If the remainders of some topic is too high, so the remain value goes there.
def weighting_allocation(n: int , prob: dict):
  exact_allo = {topic : n*p for topic, p in prob.items()}
  init_allo = {topic : int(allo) for topic, allo in exact_allo.items()}
  remain_q = n - sum(init_allo.values())
  logger_s.info(f"exact_allo: {exact_allo}")
  logger_s.info(f"init_allo: {init_allo}")
  logger_s.info(f"remain_q: {remain_q}")
  if remain_q != 0:
    remainders = {topic : val- init_allo[topic] for topic, val in exact_allo.items()}
    sort_topics = sorted(remainders, key = lambda x: x[1], reverse = True)
    logger_s.info(f"sort_topics: {sort_topics}")
    for i in range(remain_q):
      topic = sort_topics[i]
      logger_s.info(f"Adding 1 to {topic}")
      init_allo[topic] += 1  # Ensure all questions are allocated
  return init_allo


def weighting_matrix(topic_counts, level_counts):
    """
    Creates a deterministic weighting matrix ensuring the total sum matches topic_counts sum.
    
    Args:
        topic_counts (dict): Dictionary mapping topics to their required counts
        level_counts (dict): Dictionary mapping difficulty levels to their required counts
    """
    total_questions = sum(topic_counts.values())
    # Verify that level_counts sum matches total_questions
    if sum(level_counts.values()) != total_questions:
        raise ValueError(f"Sum of level_counts ({sum(level_counts.values())}) must equal sum of topic_counts ({total_questions})")
    
    level_topic_matrix = {}
    sorted_levels = sorted(level_counts.keys())
    sorted_topics = sorted(topic_counts.keys())
    
    # Initialize the matrix with zeros
    for level in sorted_levels:
        level_topic_matrix[level] = {topic: 0 for topic in sorted_topics}
    
    # First, distribute by level
    questions_placed = 0
    for level in sorted_levels:
        level_questions = level_counts[level]
        if questions_placed + level_questions > total_questions:
            level_questions = total_questions - questions_placed
            
        # Distribute proportionally across topics
        remaining = level_questions
        for topic in sorted_topics[:-1]:  # All topics except the last one
            topic_ratio = topic_counts[topic] / total_questions
            count = int(np.floor(level_questions * topic_ratio))
            count = min(count, remaining, topic_counts[topic])  # Don't exceed topic's total
            level_topic_matrix[level][topic] = count
            remaining -= count
            
        # Put remaining questions in the last topic
        if remaining > 0:
            level_topic_matrix[level][sorted_topics[-1]] = remaining
            
        questions_placed += level_questions
        
    # Adjust to match exact topic counts
    for topic in sorted_topics:
        current_sum = sum(level_topic_matrix[level][topic] for level in sorted_levels)
        diff = topic_counts[topic] - current_sum
        
        if diff > 0:
            # Add remaining questions to levels that have room
            for level in sorted_levels:
                level_sum = sum(level_topic_matrix[level].values())
                if level_sum < level_counts[level] and diff > 0:
                    add_amount = min(diff, level_counts[level] - level_sum)
                    level_topic_matrix[level][topic] += add_amount
                    diff -= add_amount
    
    # Final verification
    final_sum = sum(sum(level_dict.values()) for level_dict in level_topic_matrix.values())
    if final_sum != total_questions:
        raise ValueError(f"Final sum {final_sum} doesn't match required total {total_questions}")
        
    return level_topic_matrix
   
   

      
# Application Programming Interface
app = FastAPI()


# Add Questions API Endpoints
@app.post("/questions/add")
def add_questions(num: int = Query(..., gt=0, description="Number of questions to add"), 
                  topics: Optional[List[str]] = Query(None, description='List of topics for the questions'),
                  levels: Optional[List[str]] = Query(None, description='List of levels for the questions')): # gt = greater than 
    
  """
    API endpoint to add questions to the database.

    Args:
        num (int): Number of questions to add.
        topics (Optional[List[str]]): List of topics for question generation. If none provided, default topics are used.

    Returns:
        dict: Status message with details of the operation.
  """

  try:
      if topics is None:
        logger_aq.info("Loading default topics from file")
        try:
          with open(INIT_TOP_FILE) as f:
            topic_str = f.read()
        except FileNotFoundError as e:
          logger_aq.error(f"Error: {e}")
          return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                              detail="File not found")
        
      else:
        logger_aq.info(f"Received topics: {topics}")
        topic_str = '\n'.join([f"{i+1}. {top}" for i, top in enumerate(topics)])
        logger_aq.info(f"Topics: {topic_str}")

      if levels is None:
        levels = ['Easy', 'Medium', 'Hard']
        level_str = '\n'.join([f"{i+1}. {lvl}" for i, lvl in enumerate(levels)])
      else:
        logger_aq.info(f"Received levels: {levels}")
        level_str = '\n'.join([f"{i+1}. {lvl}" for i, lvl in enumerate(levels)])
        logger_aq.info(f"Levels: {level_str}")
        
      
      # Generate questions based on topic
      agent = Agent(TEMPLATE_FILE)
      # Batching the questions to generate

      logger_aq.info(f"Generating {num} questions in batches of {BATCH_SIZE}.")
      raw_questions = {'questions': []}
      batch_list = [BATCH_SIZE]*(num//BATCH_SIZE) + [num%BATCH_SIZE] if num > BATCH_SIZE else [num]
      for i in batch_list:
          response = agent.generate(i, topic_str, level_str)
          raw_questions['questions'].extend(response['questions'])  # extend the list

      # Encode new questions
      docs = ['\n'.join([r['question'], r['code']]) for r in raw_questions['questions']] # Encode both question and code
      logger_aq.info(f"Encoding {len(docs)} questions.")
      doc_emb = openai_model.encode_documents(docs)
      
      # Diversify the questions (sim >= 0.95 => Not Added)
      logger_aq.info("Filtering questions based on similarity threshold (>= 0.95).")
      idx_to_remove = find_idx_similarity(db, COL_NAME, doc_emb) # OUTPUT: Index of doc_emb that will be removed (not the ID of database)
      num_questions_added = len(doc_emb) - len(idx_to_remove)
      logger_aq.info(f"Number of questions added: {num_questions_added}")
      logger_aq.info(f"Number of questions removed: {len(idx_to_remove)}")
      final_questions = []
      delete_questions = []
      # Add to Database and Remove too similar questions
      for i, q in enumerate(raw_questions['questions']):
          if i not in idx_to_remove:
              final_questions.append(q)
              data = [{'embedding': doc_emb[i], 
                      'level': q['level'], 
                      'question': q['question'], 
                      'code' : q['code'],
                      'explanation': q['explanation'],
                      'opt_1': q['opt_1'], 
                      'opt_2': q['opt_2'], 
                      'opt_3': q['opt_3'], 
                      'opt_4': q['opt_4'], 
                      'correct_opt': q['correct_opt']}]
              db.insert(COL_NAME, data)
          elif i in idx_to_remove:
              delete_questions.append(q)

      return JSONResponse(status_code=status.HTTP_200_OK, 
                          content={
                            "status": "success",
                            "num_questions_added": num_questions_added,
                            "removed_similar_questions": len(idx_to_remove),
                            "questions_added" : final_questions,
                            "questions_removed" : delete_questions
                          })
  
  except HTTPException as e:
      logger_aq.error(f"Error: {e}")
      return e  
  
  except Exception as e:
      logger_aq.error(f"Error while adding questions: {e}")
      return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                           detail="Unexpected error occured: {e}")


# Search Questions API Endpoints
@app.post("/questions/search")
def search_questions(query_topics: Optional[dict] = Body(dict, description="Query topics wanted to search questions"),
                     query_level: Optional[dict] = Body(dict, description="Query level wanted to search questions"),
                     num: int = Query(..., gt=0, description="Number of questions wanted")):
  """
    API endpoint to search questions based on a query topic.

    Args:
        query_topics (Optional[dict]): Query topics wanted to search questions.  If don't want to specify, just used "any topics"
                  EX. {'0' : 'Pandas', '1' : 'Numpy', '2' : 'Matplotlib'}
        query_level (Optional[dict]): Query level wanted to search questions.
                  EX. {'0' : 'Easy', '1' : 'Medium', '2' : 'Hard'}
        num (int): Number of questions wanted.
    
    Returns:
        dict: List of questions found based on the query.
    
        
  """
  try:
      tot_in_bank = db.get_collection_stats(collection_name=COL_NAME)['row_count']
      if num > tot_in_bank:
         return HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                              detail="Number of questions requested is greater than the total in the question bank.")
      
      # Inverse Rank weighting distribution = 1/(1+rank)  *Rank start with 0
      # Calculate question distribution
      logger_s.info("Calculate question distribution")
      q_weights = {topic : 1 / (1 + int(rank)) for rank, topic in query_topics.items()} 
      logger_s.info(f"Question count: {q_weights}")
      q_prob_dist = {topic : weight/sum(q_weights.values()) for topic, weight in q_weights.items()}
      logger_s.info(f"Question probability distribution: {q_prob_dist}")
      question_count = weighting_allocation(num, q_prob_dist)
      

      # Calculate level distribution
      logger_s.info("Calculate level distribution")
      level_weights = {level : 1 / (1 + int(rank)) for rank, level in query_level.items()}
      level_prob_dist = {level : weight/sum(level_weights.values()) for level, weight in level_weights.items()}
      level_count = weighting_allocation(num, level_prob_dist)

      # Calculate the matrix
      logger_s.info("Calculate the matrix")
      logger_s.info(f"Question count: {question_count}")
      logger_s.info(f"Level count: {level_count}")
      level_topic_matrix = weighting_matrix(question_count, level_count)
      logger_s.info(f"Level-Topic Matrix: {level_topic_matrix}")
      # Search for questions based on query topics
      query_template = """What questions is about {}"""
      logger_s.info("Searching questions based on query topics")
      result_questions = []
      idx_selected = []
      tot_miss_q = 0
      for lvl in level_topic_matrix:
        for topic in level_topic_matrix[lvl]:
          if level_topic_matrix[lvl][topic] == 0:
            continue
          # Search the questions
          full_query = query_template.format(topic)

          # Embedding full_query
          query_emb = openai_model.encode_documents([full_query])

          logger_s.info(f"Searching questions for {lvl} level and {topic} topic")
          res = db.search(collection_name=COL_NAME, 
                          data = query_emb, 
                          output_fields = ['question', 'code', 'opt_1', 'opt_2', 'opt_3', 'opt_4', 'correct_opt', 'level', 'explanation'],
                          limit = level_topic_matrix[lvl][topic]*ALPHA,  # DOESN"T NEED TO USE LIMIT, IF USED RERANKING (USED THERE INSTEAD)
                          ) # IF HAVE FILTER: filter = f'level == \"{lvl}\" and id not in {idx_selected}'
          
          # Reranking
          doc = [res[0][i]['entity']['question'] for i in range(len(res[0]))]

          rr_result = jina_rf(
                  query = full_query,
                  documents = doc,
                  top_k = level_topic_matrix[lvl][topic]
          )

          idx_selected_rr = [r.index for r in rr_result]
          final_set_q = [res[0][i] for i in idx_selected_rr if res[0][i]['id'] not in idx_selected and res[0][i]['entity']['level'] == lvl] # Add each question (in json format) to the final set given it's the one in top_k of reranker

          result_questions.extend(final_set_q)
          idx_selected.extend([r['id'] for r in final_set_q])

          # If have missing questions
          missing_questions = level_topic_matrix[lvl][topic] - len(final_set_q)
          tot_miss_q += missing_questions
          if missing_questions > 0:
            logger_s.info(f"Missing questions for {lvl} level and {topic} topic: {missing_questions} questions")
          
      
      # query for missing questions (Technically, it's ordered from IDs that hasn't yet be selected. The topics may be varied)
      if tot_miss_q > 0:
        logger_s.info(f"Total missing questions: {tot_miss_q}")

        if tot_miss_q == num: # No questions is able to be retrieved
          miss_res = db.query(collection_name=COL_NAME, 
                            output_fields = ['question', 'code', 'opt_1', 'opt_2', 'opt_3', 'opt_4', 'correct_opt', 'level', 'explanation'],
                            limit = tot_miss_q)
          result_questions.extend(miss_res)
        else:
          miss_res = db.query(collection_name=COL_NAME, 
                              output_fields = ['question', 'code', 'opt_1', 'opt_2', 'opt_3', 'opt_4', 'correct_opt', 'level', 'explanation'],
                              limit = tot_miss_q,
                              filter= f'id not in {idx_selected}')
          result_questions.extend(miss_res)

      return JSONResponse(status_code=status.HTTP_200_OK, 
                          content={
                            "status": "success",
                            "questions": result_questions
                          })
  
  except HTTPException as e:
      logger_s.error(f"Error: {e}")
      return e  
  
  except Exception as e:
      logger_s.error(f"Error while searching questions: {e}")
      return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                           detail="Unexpected error occured: {e}")


# Update API Endpoints
@app.put("/questions/update")
def update_question(question_id: int, 
                    topics: Optional[str] = Query(str, description="A single topic of the updated question"),
                    levels: Optional[str] = Query(str, description="A single level of the updated question")):
  """
    API endpoint to update a question in the database.
    Args:
        question_id (int): ID of the question to update.
        topics (Optional[str]): Topic of the updated question.

    Returns:
        dict: Status message with details of the operation.
  """
   # Upsert (Update OR Insert)
  try:
    num = 1
    if topics is None:
      logger_aq.info("Loading default topics from file")
      try:
        with open(INIT_TOP_FILE) as f:
          topic_str = f.read()
      except FileNotFoundError as e:
        logger_aq.error(f"Error: {e}")
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail="File not found")
      
    else:
      logger_aq.info(f"Received topics: {topics}")
      topic_str = '\n'.join([f"{i+1}. {top}" for i, top in enumerate([topics])])

    if levels is None:
      levels = ['Easy', 'Medium', 'Hard']
      level_str = '\n'.join([f"{i+1}. {lvl}" for i, lvl in enumerate([levels])])
    else:
      logger_aq.info(f"Received levels: {levels}")
      level_str = '\n'.join([f"{i+1}. {lvl}" for i, lvl in enumerate([levels])])
      logger_aq.info(f"Levels: {level_str}")
    
    # Generate questions based on topic
    agent = Agent(TEMPLATE_FILE)
    raw_questions = {'questions': []}

    response = agent.generate(num, topic_str, level_str)
    raw_questions['questions'].extend(response['questions'])  # extend the list

    # Encode new questions
    docs = ['\n'.join([r['question'], r['code']]) for r in raw_questions['questions']] # Encode both question and code
    logger_aq.info(f"Encoding {len(docs)} questions.")
    doc_emb = openai_model.encode_documents(docs)

    # Upsert to database
    logger_aq.info("Upserting questions to the database.")
    data = [{'id': question_id, 
            'embedding': doc_emb[i], 
            'level': q['level'], 
            'question': q['question'], 
            'code' : q['code'],
            'explanation': q['explanation'],
            'opt_1': q['opt_1'], 
            'opt_2': q['opt_2'], 
            'opt_3': q['opt_3'], 
            'opt_4': q['opt_4'], 
            'correct_opt': q['correct_opt']} for i, q in enumerate(raw_questions['questions'])]
    db.upsert(COL_NAME, data)

    return JSONResponse(status_code=status.HTTP_200_OK, 
                        content={
                          "status": "success",
                          "question_idx_replace": question_id,
                          "new_question": raw_questions['questions']
                        })
  
  except HTTPException as e:
    logger_aq.error(f"Error: {e}")
    return e
  
  except Exception as e:
    logger_aq.error(f"Error while updating questions: {e}")
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                         detail="Unexpected error occured: {e}")

# Delete API Endpoints
@app.delete("/questions/delete")
def delete_question(question_id: int):
  """
    API endpoint to delete a question from the database.

    Args:
        question_id (int): ID of the question to delete.

    Returns:
        dict: Status message with details of the operation.
  """
  try:
    db.delete(COL_NAME, [question_id])
    return JSONResponse(status_code=status.HTTP_200_OK, 
                        content={
                          "status": "success",
                          "question_idx_deleted": question_id
                        })
  
  except HTTPException as e:
    logger_aq.error(f"Error: {e}")
    return e
  
  except Exception as e:
    logger_aq.error(f"Error while deleting questions: {e}")
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                         detail="Unexpected error occured: {e}")

    

