import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NOTE:  For MacOS for sure

    One important tip for Python 3.13

    Some CLIP dependencies (especially torch) can still be spotty with 3.13 depending on platform.

    If you hit strange install errors, the most stable setup is:

    * Python 3.11 or 3.12
    * embetter[recommended]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 Class Image Classification comparing different classification techniques
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook will look at the following techniques for performing image classification:

    * Keras/Tensorflow
    * CLIP Image Embeddings/Scikit-Learn
    * Zero Shot Image Classification with CLIP Image and Text Embeddings
    * Vision Language Models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The impetus for this notebook started by taking the course, "Deep Learning with Tensorflow & Keras" from opencv.org.

    https://opencv.org/university/deep-learning-with-tensorflow-keras/?utm_source=lopcv&utm_medium=menu

    One of the course assignments was to create a custom Keras CNN model to perform 4 animal image classification.  The animals were

    * cow
    * elephant
    * horse
    * spider

    In the assignment you had to reach at least 80% accuracy to pass the assignment.

    Since taking that class, embedding models and Vision Language Models have come onto the scene so I wanted to revist that assignment with the new techniques to see how they compared.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook will start with the 4 animals dataset that was presented in the course, and later in the notebook we will add cats and dogs to the dataset.

    Each of the techniques will create a submission.csv just like we needed for the course, and the submission will be 'graded' locally against the true submission.  In the course you are never given the true submission values but you can easily derive this by creating your own from the images in the Test directory
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Keras / Tensorflow
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I am not going to revist the actual Keras CNN model code but go over my results as the baseline to see how the other techniques compare.

    Developing the Keras CNN model and training took hours to run.  Granted I should have used Google Colab or another cloud service instead of my Mac.  Regardless, developing the model architecture and training took almost a week of my time, and lots of local compute time.

    After all of that, my best accuracy was **0.88292**

    The best overall accuracy was **0.97560**
    """)
    return


@app.cell(hide_code=True)
def _(mo):

    mo.image("media/kerastf_summary_board.png", caption="Keras/TF 4 Class Classification Summary")
    return


@app.cell
def _(mo):
    mo.image("media/leaderboard.png", caption="Keras/TF 4 Class Classification Leaderboard")
    return


@app.cell
def _(mo):
    mo.image("media/myrank.png", caption="My Competition Rank")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CLIP ( Contrastive Language-Image Pretraining)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What is a CLIP Encoder?

    CLIP (Contrastive Language–Image Pretraining) is a model architecture designed to connect images and natural language in a shared representation space. It was introduced by OpenAI in the paper *Learning Transferable Visual Models From Natural Language Supervision*.

    For more detail here is a link to Medium article.

    https://medium.com/data-science/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2

    CLIP converts images and text into vectors that encode semantic meaning so that related images and descriptions are located near each other in the same high-dimensional space.

    Essentially, if you show CLIP and image of a dog and the word 'dog' it will know they go together.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What CLIP Does (High-Level)

    CLIP learns to answer the question:

    “Which text description best matches this image?”

    Instead of training a model for a single task (like classifying cats vs dogs), CLIP learns a general visual-language understanding.

    It does this by training on millions of (image, caption) pairs from the internet.

    Example training pair:

    Image: photo of a golden retriever running in a park

    Text: "a dog running in grass"

    CLIP learns to associate them.
    The result is a model that can:

    * Recognize objects
    * Understand visual scenes
    * Match images to text descriptions
    * Perform zero-shot classification (no task-specific training)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Core Idea: Two Encoders

    CLIP contains two neural networks:

    **Image Encoder**

    Converts an image into a numeric vector.

    Typical architectures:

    * Vision Transformer (ViT)
    * ResNet


    **Text Encoder**

    Converts text into a numeric vector.

    Typical architecture:

    Transformer (similar to GPT/BERT style models)

    Both encoders output vectors in the same vector space.

    ```
    image → image_encoder → 512 element vector/embedding
    text  → text_encoder  → 512 element vector/embedding
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Embedding

    An embedding is simply a dense numeric vector representing semantic meaning.

    Example:

    image_embedding =
    [0.021, -0.433, 0.812, 0.155, ... 512 numbers total]

    Typical CLIP embedding sizes:

    | Model | Dimensions |
    |---|---|
    | ViT-B/32 | 512 |
    | ViT-L/14 | 768 |

    * Larger models	up to 1024+

    The model that we use in this notebook, is the ViT-B/32 producing a vector of 512 elemenets.

    These vectors are L2 normalized so they lie on a unit hypersphere, meaning they are ready to be used by machine learning models that require features be normalized.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the Embedding Represents

    The embedding represents semantic concepts extracted from the image or text.

    If you had an image of a dog running in grass, instead of explicitly storing:

    * dog = 1
    * grass = 1
    * running = 1

    CLIP stores distributed meaning across the 512 dimensions.

    Conceptually, imagine the vector space with the following:

    * dimension 17 → "animalness"
    * dimension 204 → "outdoor scenes"
    * dimension 311 → "motion"

    The above is just meant to be a conceptual example of what each dimension, or index, in the embedding might mean.  When using image models with higher dimensions you can get more semantic meaning about the image.

    So the embedding might encode:

    Image: dog running in grass

    embedding = animal + outdoor + motion + mammal + pet
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How CLIP Learns This Space

    Training uses contrastive learning.

    For a batch of N image/text pairs:

    * (image1, text1)
    * (image2, text2)
    * (image3, text3)

    ...

    CLIP learns:

    * similarity(image1, text1) → HIGH
    * similarity(image1, text2) → LOW
    * similarity(image1, text3) → LOW

    Mathematically this is done using cosine similarity:

    similarity = dot(image_embedding, text_embedding)

    Training objective:

    * maximize similarity of correct pairs
    * minimize similarity of incorrect pairs

    Over millions of examples, the model organizes the embedding space so that:

    * matching images + text → close together
    * unrelated pairs → far apart
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zero-Shot Classification

    No training needed.

    image_embedding = encode(image)

    label1 = encode("a photo of a dog")

    label2 = encode("a photo of a car")

    label3 = encode("a photo of a bee")

    choose highest similarity

    CLIP can classify without ever seeing the dataset beforehand.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why This Is Useful in Machine Learning Pipelines

    CLIP embeddings can be used as general-purpose features.

    Example workflow:

    ```
    image
      ↓
    CLIP encoder
      ↓
    embedding vector
      ↓
    scikit-learn model
      ↓
    classification / clustering / anomaly detection
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CLIP Embeddings and Scikit-Learn

    This section will cover the machine learning pipeline described above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Embetter Python Package

    This section will use the Python package called, embetter.  You can get more information at the github repo.

    https://github.com/koaning/embetter

    This python package makes integrating CLIP encodings and embeddings into the Scikit-Learn machine learning pipeline.
    """)
    return


@app.cell
def _():
    from embetter.vision import ImageLoader
    from embetter.grab import ColumnGrabber
    # from embetter.multi import ClipEncoder

    from sklearn.pipeline import make_pipeline
    import umap
    from pathlib import Path
    from typing import Literal, List
    import pandas as pd

    return ColumnGrabber, ImageLoader, List, Literal, Path, make_pipeline, pd


@app.cell
def _(Path):
    target_dirs = ['cow', 'elephant', 'horse', 'spider']
    root_dir = str(Path(".").parent.resolve()) + "/dataset"
    root_dir
    return root_dir, target_dirs


@app.cell
def _():
    from embetter.multi import ClipEncoder
    ClipEncoder()
    return (ClipEncoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create a dataframe that includes the filepath to the images and the target label derived from the folder name.  The filepath will be used by embetter, to read the images and create the embeddings.
    """)
    return


@app.cell
def _(List, Literal, Path, pd, root_dir, target_dirs):
    def create_filepaths_df(dir_name: Literal['Train','Valid'], dirs: List = target_dirs) -> pd.DataFrame:
        data = []
        for dir in dirs:
            for file in Path(f'{root_dir}/{dir_name}/{dir}').glob('*.jpg'):
                row_data = {
                    'filepath': file,
                    'target': dir
                }
                data.append(row_data)
        files_df = pd.DataFrame(data, columns=["filepath", "target"])
        return files_df

    return (create_filepaths_df,)


@app.cell
def _(create_filepaths_df):
    training_files_df = create_filepaths_df(dir_name='Train')
    return (training_files_df,)


@app.cell
def _(training_files_df):
    training_files_df.sample(frac=0.006)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create a Scikit-Learn pipeline, using embetter to read the filepath column from the training dataframe, loading the image and then running each image through the ClipEncoder to generate embeddings for each image.
    """)
    return


@app.cell
def _(ClipEncoder, ColumnGrabber, ImageLoader, make_pipeline):
    # create pipeline to read the filepath column, load the image, and encode the image
    image_embedding_pipeline = make_pipeline(
       ColumnGrabber("filepath"),
      ImageLoader(convert="RGB"),
      ClipEncoder(),
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
