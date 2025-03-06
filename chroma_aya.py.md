# Langchain Chroma Vectorstore Documentation

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. `Chroma_AYA` Class](#2-chroma_aya-class)
    * [2.1 `__init__` Method](#21-init-method)
    * [2.2 `embeddings` Property](#22-embeddings-property)
    * [2.3 `__query_collection` Method](#23-query_collection-method)
    * [2.4 `encode_image` Method](#24-encode_image-method)
    * [2.5 `add_images` Method](#25-add_images-method)
    * [2.6 `add_texts` Method](#26-add_texts-method)
    * [2.7 `similarity_search` Method](#27-similarity_search-method)
    * [2.8 `similarity_search_by_vector` Method](#28-similarity_search_by_vector-method)
    * [2.9 `similarity_search_by_vector_with_relevance_scores` Method](#29-similarity_search_by_vector_with_relevance_scores-method)
    * [2.10 `similarity_search_with_score` Method](#210-similarity_search_with_score-method)
    * [2.11 `_select_relevance_score_fn` Method](#211-_select_relevance_score_fn-method)
    * [2.12 `max_marginal_relevance_search_by_vector` Method](#212-max_marginal_relevance_search_by_vector-method)
    * [2.13 `max_marginal_relevance_search` Method](#213-max_marginal_relevance_search-method)
    * [2.14 `delete_collection` Method](#214-delete_collection-method)
    * [2.15 `get` Method](#215-get-method)
    * [2.16 `persist` Method](#216-persist-method)
    * [2.17 `update_document` Method](#217-update_document-method)
    * [2.18 `update_documents` Method](#218-update_documents-method)
    * [2.19 `from_texts` Class Method](#219-from_texts-class-method)
    * [2.20 `from_documents` Class Method](#220-from_documents-class-method)
    * [2.21 `delete` Method](#221-delete-method)
    * [2.22 `__len__` Method](#222-len-method)
* [3. Helper Functions](#3-helper-functions)
    * [3.1 `_results_to_docs` Function](#31-_results_to_docs-function)
    * [3.2 `_results_to_docs_and_scores` Function](#32-_results_to_docs_and_scores-function)


<a name="1-introduction"></a>
## 1. Introduction

This document provides internal code documentation for the `Chroma_AYA` class, a Langchain vectorstore built on top of ChromaDB.  It details the functionality and implementation of its methods, focusing on the algorithms and logic behind more complex operations.


<a name="2-chroma_aya-class"></a>
## 2. `Chroma_AYA` Class

This class provides an interface for interacting with a ChromaDB vector database.


<a name="21-init-method"></a>
### 2.1 `__init__` Method

The constructor initializes the `Chroma_AYA` object. It handles different initialization scenarios, including:

* **Client Provided:** If a ChromaDB client is provided, it's directly used.
* **`client_settings` Provided:** If `client_settings` are provided, a client is created using these settings.  Backwards compatibility for ChromaDB versions below 0.4.0 is included by setting the `chroma_db_impl` to `"duckdb+parquet"`.  If `persist_directory` is also provided, the collection is persisted to disk.
* **`persist_directory` Provided:** If only a `persist_directory` is provided, a client is created with persistence enabled. Backwards compatibility for ChromaDB versions below 0.4.0 is maintained.
* **No Persistence:** If neither `client_settings` nor `persist_directory` is provided, an in-memory client is created.

The constructor also initializes the embedding function, retrieves or creates the specified ChromaDB collection, and sets a relevance score function (optional override).


<a name="22-embeddings-property"></a>
### 2.2 `embeddings` Property

A getter for the embedding function used by the vectorstore.


<a name="23-query_collection-method"></a>
### 2.3 `__query_collection` Method

This method performs a query on the underlying ChromaDB collection. It accepts either `query_texts` or `query_embeddings` as input, along with parameters for controlling the number of results (`n_results`), filtering (`where`, `where_document`), and additional keyword arguments.  It returns a dictionary containing query results from ChromaDB.


<a name="24-encode_image-method"></a>
### 2.4 `encode_image` Method

Encodes an image from a given URI into a base64 string.  This is used for image storage in ChromaDB.


<a name="25-add_images-method"></a>
### 2.5 `add_images` Method

Adds images to the ChromaDB collection.  It handles:

1. **Base64 Encoding:** Encodes image URIs to base64 strings.
2. **ID Generation:** Generates UUIDs if IDs are not provided.
3. **Embedding:** If an embedding function and the `embed_image` method are available, it generates embeddings for the images.
4. **Metadata Handling:** It intelligently manages metadata, handling cases where metadata is provided for only a subset of images. It also includes error handling for complex metadata values, suggesting the use of `filter_complex_metadata` if a `ValueError` occurs.
5. **Upserting:**  It uses ChromaDB's `upsert` method to add or update the images in the collection, splitting the operation into batches for images with and without metadata for efficient handling of potentially complex metadata.


<a name="26-add_texts-method"></a>
### 2.6 `add_texts` Method

Adds texts to the ChromaDB collection.  Similar to `add_images`, it handles ID generation, embedding (if an embedding function is provided), metadata management (including error handling and suggestions for `filter_complex_metadata`), and upserting using ChromaDB's `upsert` method.  It also splits the upsert into batches for texts with and without metadata.


<a name="27-similarity_search-method"></a>
### 2.7 `similarity_search` Method

Performs a similarity search using text queries. It uses the `similarity_search_with_score` method internally and returns only the documents, discarding the scores.


<a name="28-similarity_search-by-vector-method"></a>
### 2.8 `similarity_search_by_vector` Method

Performs a similarity search using embedding vectors.  It directly calls `__query_collection` with the provided embedding and returns the documents using the `_results_to_docs` helper function.


<a name="29-similarity_search_by_vector_with_relevance_scores-method"></a>
### 2.9 `similarity_search_by_vector_with_relevance_scores` Method

Performs a similarity search using embedding vectors and returns both the documents and their similarity scores.  It uses `__query_collection` and `_results_to_docs_and_scores`.


<a name="210-similarity_search_with_score-method"></a>
### 2.10 `similarity_search_with_score` Method

Performs a similarity search using text queries and returns both the documents and their similarity scores.  If an embedding function is available, it generates the query embedding before calling `__query_collection`.  Otherwise, it directly uses the query text. The results are processed using `_results_to_docs_and_scores`.


<a name="211-_select_relevance_score_fn-method"></a>
### 2.11 `_select_relevance_score_fn` Method

Selects an appropriate relevance score function based on the distance metric used by the ChromaDB collection. It checks the collection metadata for the `hnsw:space` key to determine the distance metric. Currently supports "cosine", "l2" (Euclidean), and "ip" (inner product). It prioritizes a user-provided `relevance_score_fn` if available.  If no appropriate function is found, a `ValueError` is raised.


<a name="212-max_marginal_relevance_search_by_vector-method"></a>
### 2.12 `max_marginal_relevance_search_by_vector` Method

Performs a Maximal Marginal Relevance (MMR) search using embedding vectors.  It fetches a larger set of candidate documents (`fetch_k`) using `__query_collection`, then applies the MMR algorithm from `langchain_community.vectorstores.utils` to select the top `k` documents that balance relevance and diversity.


<a name="213-max_marginal_relevance_search-method"></a>
### 2.13 `max_marginal_relevance_search` Method

Performs an MMR search using a text query. It first generates the query embedding using the embedding function and then calls `max_marginal_relevance_search_by_vector`.  A `ValueError` is raised if no embedding function is available.


<a name="214-delete_collection-method"></a>
### 2.14 `delete_collection` Method

Deletes the ChromaDB collection.


<a name="215-get-method"></a>
### 2.15 `get` Method

Retrieves documents from the collection based on various criteria, including IDs, where clauses, limits, offsets, and included fields (`embeddings`, `metadatas`, `documents`).  It uses the ChromaDB collection's `get` method.


<a name="216-persist-method"></a>
### 2.16 `persist` Method

Persists the ChromaDB collection to disk. This method is deprecated since Chroma 0.4.x, as persistence is now handled automatically.


<a name="217-update_document-method"></a>
### 2.17 `update_document` Method

Updates a single document in the collection. It uses the `update_documents` method internally.


<a name="218-update_documents-method"></a>
### 2.18 `update_documents` Method

Updates multiple documents in the collection. It generates embeddings if an embedding function is available, then uses ChromaDB's `update` method, handling batching for Chroma versions 0.4.10 and above using `create_batches` from `chromadb.utils.batch_utils`.


<a name="219-from_texts-class-method"></a>
### 2.19 `from_texts` Class Method

Creates a `Chroma_AYA` object from a list of texts. It handles embedding generation, ID generation, and adding the texts to the collection using `add_texts`. It supports batching for Chroma versions 0.4.10 and above using `create_batches`.


<a name="220-from_documents-class-method"></a>
### 2.20 `from_documents` Class Method

Creates a `Chroma_AYA` object from a list of `Document` objects. It extracts texts and metadatas from the documents and calls `from_texts` to create the vectorstore.


<a name="221-delete-method"></a>
### 2.21 `delete` Method

Deletes documents from the collection based on their IDs.


<a name="222-len-method"></a>
### 2.22 `__len__` Method

Returns the number of documents in the collection using ChromaDB's `count` method.


<a name="3-helper-functions"></a>
## 3. Helper Functions


<a name="31-_results_to_docs-function"></a>
### 3.1 `_results_to_docs` Function

This helper function converts the results from a ChromaDB query into a list of `Document` objects, extracting only the document content.


<a name="32-_results_to_docs_and_scores-function"></a>
### 3.2 `_results_to_docs_and_scores` Function

This helper function converts the results from a ChromaDB query into a list of tuples, where each tuple contains a `Document` object and its corresponding similarity score.  It extracts document content, metadata, and distance scores from the query results.  The `id` is added to the metadata. There is a TODO comment flagging that the current implementation only considers the first result of a batch query, indicating potential future improvements for batch processing.
