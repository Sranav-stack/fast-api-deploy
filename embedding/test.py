from bgme import OpenRouterBGEEmbeddingFunction

def test_embedding_function():
    # Initialize the embedding function
    embedding_function = OpenRouterBGEEmbeddingFunction()

    # Sample input string
    documents = [
        "This is the first test document.",
        "Here is another example of a document.",
        "Testing embeddings with a third document."
    ]

    # Generate embeddings
    embeddings = embedding_function(documents)

    # Print the embeddings
    print("Generated Embeddings:")
    for i, embedding in enumerate(embeddings):
        print(f"Document {i + 1}: {embedding[:5]}...")  # Print first 5 values for brevity

if __name__ == "__main__":
    test_embedding_function()