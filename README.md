# Custom NER model using Spacy
A custom NER (Named Entity Recognition) model can be built using Spacy, a popular natural language processing library. The process involves training the model on labeled data to recognize specific entities like names, organizations, and locations. Firstly, annotate a      training dataset with entity labels. Then, train the Spacy model using the annotated data. Fine-tune the model by iterating over the training process. Finally, evaluate the model's performance and save it for future use. Spacy's flexible architecture and pre-built components make it a powerful tool for creating custom NER models tailored to specific entity recognition tasks.

# Joining entity and relationship extraction models
Joining entity and relationship extraction models using a combination of BERT Transformer and Spacy can yield powerful results for natural language processing tasks. BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art language model known for its contextual word representations. Spacy, on the other hand, offers robust tokenization, part-of-speech tagging, and dependency parsing capabilities. To build the joint entity and relationship extraction model, we can follow a two-step process. First, we leverage BERT to encode the input text and obtain contextualized word representations. These representations capture the semantic meaning and context of the words. Next, we utilize Spacy's dependency parsing to extract entities and their relationships based on the encoded representations. Spacy's dependency parsing helps identify the syntactic relationships between words, which can be used to determine the relationships between entities. By combining the strengths of BERT and Spacy, we can enhance both entity recognition and relationship extraction. The contextualized word representations from BERT enable better entity recognition by capturing the surrounding context, while Spacy's dependency parsing helps identify the connections and relationships between entities. This joint model can be trained using labeled data, where entities are annotated along with their corresponding relationships. Fine-tuning can be performed iteratively to optimize the model's performance. Overall, the combination of BERT Transformer and Spacy provides a powerful framework for extracting entities and relationships from text, enabling applications such as information extraction, knowledge graph construction, and question answering systems.

# UBIAI annotator 
UBIAI Annotator is an advanced text annotation tool that simplifies the process of creating labeled datasets for machine learning tasks. It offers a user-friendly interface for annotating entities, relationships, and other relevant information in text data. With UBIAI Annotator, users can quickly annotate large volumes of text, collaborate with team members, and maintain consistency in labeling. The tool supports various annotation types, such as named entity recognition (NER), entity linking, and sentiment analysis. It also provides features like automatic suggestions, customizable annotation templates, and easy export of annotated data. UBIAI Annotator streamlines the annotation workflow, making it an invaluable asset for training and evaluating machine learning models.

# Developer: Dr. Partha Majumder
# Email: parthamajpk@gmail.com

# License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg




