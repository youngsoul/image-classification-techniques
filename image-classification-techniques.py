# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "embetter[sbert]>=0.7.0,<0.8.0",
#   "jupyter>=1.1.1",
#   "marimo[recommended]>=0.21.0",
#   "pillow>=12.1.1",
#   "scikit-learn>=1.8.0",
#   "umap-learn>=0.5.11",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import altair as alt
    from embetter.vision import ImageLoader
    from embetter.grab import ColumnGrabber
    from embetter.multi import ClipEncoder

    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics.pairwise import cosine_similarity
    import os
    import umap
    from pathlib import Path
    from typing import Literal, List
    import pandas as pd
    import numpy as np
    import ollama
    import sys
    import csv


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # NOTE:  Embetter requires install of Jupyter when running in marimo notebook.

    I do not know why yet, but to get the embetter python package to run in a marimo notebook, I had to install jupyter.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## TL;DR Summary

    Comparing 4 techniques for 4 animal image classification

    * Keras/Tensorflow
      - My personal best accuracy was 0.88292.  The best course accuracy was 0.97560.


    * CLIP Image Embeddings/Scikit-Learn
        - Accuracy: 0.9963

    * Zero Shot Image Classification with CLIP Image and Text Embeddings
      - Accuracy: 0.9963

    * Vision Language Models
      - Accuracy: 1.000

    Back in the day when CNN models were primarily used for image classification, it took significant time and compute resources.  The Keras CNN models also performed the worst compared to current techniques.

    Using CLIP embeddings either with Scikit-Learn or as Zero-Shot classifier performed very well and were fast to build, train and execute.

    The best accuracy was from using Vision Language Models.  The drawback to this approach is the speed to classifiy all of the test submission images.  This step is very compute dependent.

    If accuracy is the key metric, and speed of inference is not a factor then using VMLs is the way to go.

    If speed of inference is the key metric, and you can live with less than perfect performance either Scikit-Learn or Zero-Shot is the way to go.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4 Class Image Classification comparing different classification techniques
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This notebook will look at the following techniques for performing image classification:

    * Keras/Tensorflow
    * CLIP Image Embeddings/Scikit-Learn
    * Zero Shot Image Classification with CLIP Image and Text Embeddings
    * Vision Language Models
    """)
    return


@app.cell(hide_code=True)
def _():
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
def _():
    mo.md(r"""
    This notebook will start with the 4 animals dataset that was presented in the course, and later in the notebook we will add cats and dogs to the dataset.

    Each of the techniques will create a submission.csv just like we needed for the course, and the submission will be 'graded' locally against the true submission.  In the course you are never given the true submission values but you can easily derive this by creating your own from the images in the Test directory
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Keras / Tensorflow
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    I am not going to revist the actual Keras CNN model code but go over my results as the baseline to see how the other techniques compare.

    Developing the Keras CNN model and training took hours to run.  Granted I should have used Google Colab or another cloud service instead of my Mac.  Regardless, developing the model architecture and training took almost a week of my time, and lots of local compute time.

    After all of that, my best accuracy was **0.88292**

    The best overall accuracy was **0.97560**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image("media/kerastf_summary_board.png", caption="Keras/TF 4 Class Classification Summary")
    return


@app.cell
def _():
    mo.image("media/leaderboard.png", caption="Keras/TF 4 Class Classification Leaderboard")
    return


@app.cell
def _():
    mo.image("media/myrank.png", caption="My Competition Rank")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## CLIP ( Contrastive Language-Image Pretraining)
    """)
    return


@app.cell(hide_code=True)
def _():
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
def _():
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
def _():
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
def _():
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
def _():
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
def _():
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
def _():
    mo.md(r"""
    ### Zero-Shot Classification

    No training needed.

    image_embedding = encode(image)

    dog_label_embedding = encode("a photo of a dog")

    car_label_embedding = encode("a photo of a car")

    bee_label_embedding = encode("a photo of a bee")

    Compare each of the label embeddings, and choose highest similarity label as the prediction.

    CLIP can classify without ever seeing the dataset beforehand.
    """)
    return


@app.cell(hide_code=True)
def _():
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
def _():
    mo.md(r"""
    ## CLIP Embeddings and Scikit-Learn

    This section will cover the machine learning pipeline using ClipEncoder to generate image embeddings, then us Scikit-Learn LogitistRegression to perform image classification.

    ```
    image
      ↓
    CLIP encoder
      ↓
    image embedding vector
      ↓
    scikit-learn LogisticRegression model
      ↓
    classification
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Embetter Python Package

    This section will use the Python package called, **embetter**.  You can get more information at the github repo.

    https://github.com/koaning/embetter

    This python package makes integrating CLIP encodings and embeddings into the Scikit-Learn machine learning pipeline.
    """)
    return


@app.cell
def _():
    target_dirs = ['cow', 'elephant', 'horse', 'spider']
    root_dir = str(Path(".").parent.resolve()) + "/dataset"
    root_dir
    return root_dir, target_dirs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Create a dataframe that includes the filepath to the images and the target label derived from the folder name.  The filepath will be used by embetter, to read the images and create the embeddings.
    """)
    return


@app.cell
def _(root_dir, target_dirs):
    def create_filepaths_df(dir_name: Literal['Train','Valid', 'Test'], dirs: List = target_dirs) -> pd.DataFrame:
        """ create a dataframe with columns filepath and target.  the filepath column will contain full qualified path to the image files in the dataset train/valid directoroes.  The target column will contain the target name of the image such as spider, horse, etc."""
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
def _():
    dropdown = mo.ui.dropdown(
        options=["Train", "Valid"], value="Train", label="Choose Training or Validation images to inspect"
    )
    dropdown
    return (dropdown,)


@app.cell
def _(create_filepaths_df, dropdown):
    image_files_df = create_filepaths_df(dir_name=f'{dropdown.value}')
    image_files_df.shape
    return (image_files_df,)


@app.cell
def _(image_files_df):
    image_files_df.sample(frac=0.006).assign(
        filepath=lambda df: df["filepath"].map(lambda p: Path(str(p)).name)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The file path column is a fully qualified path to the image.  For clarity I am just showing the filename.

    Create a Scikit-Learn pipeline, using the embetter Python package to read the filepath column from the training dataframe, loading the image and then running each image through the ClipEncoder to generate embeddings for each image.
    """)
    return


@app.cell
def _():
    # create pipeline to read the filepath column, load the image, and encode the image
    image_clip_encoder = ClipEncoder()
    image_embedding_pipeline = make_pipeline(
       ColumnGrabber("filepath"),
       ImageLoader(convert="RGB"),
       image_clip_encoder,
    )
    return image_clip_encoder, image_embedding_pipeline


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The image_embedding_pipeline will use the image_files_df dataframe, first looking for a column with the name, 'filepath'.  For every value in the 'filepath' column, the ImageLoader will load the image file and run the image through the ClipEncoder to produce a 512 element embedding vector for each image.
    """)
    return


@app.cell
def _(image_embedding_pipeline, image_files_df):
    # convert the filepaths to embeddings
    X = image_embedding_pipeline.fit_transform(image_files_df)
    return (X,)


@app.cell
def _(X):
    print(type(X))
    print(X.shape)
    return


@app.cell
def _(image_files_df):
    y = image_files_df['target']

    print(y.shape)
    return (y,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    At this point we have a very traditional dataset with 512 features and a column of target values.  We can then use this dataset with Scikit-Learn to train a classification model.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Visualize the embedding space with UMAP dimensionality reduction

    Because we now have a vector representation of each image, we can reduce the dimensionality to 2 dimensions and plot each image.  Hopefully what we see is a clustering of like images.  Cow images should cluster with other cow images, spider images should cluster with other spider images.
    """)
    return


@app.cell
def _():
    reducer = umap.UMAP(n_components=2, random_state=42)
    return (reducer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Reduce the 512 dimensional embedding vectors to 2 dimensions
    """)
    return


@app.cell
def _(X, reducer):
    reduced_embeddings = reducer.fit_transform(X)
    return (reduced_embeddings,)


@app.cell
def _(X, reduced_embeddings):
    print(f"Original shape: {X.shape}")
    print(f"Reduced shape: {reduced_embeddings.shape}")
    print(type(reduced_embeddings))
    return


@app.cell
def _(reduced_embeddings, y):
    reduced_embeddings_df = pd.DataFrame(
        {
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "target": y
        }
    )
    return (reduced_embeddings_df,)


@app.cell
def _(reduced_embeddings_df):

    selection = alt.selection_point(fields=["target"], bind="legend")
    chart = (
        alt.Chart(reduced_embeddings_df)
        .mark_point()
        .encode(
            x="x",
            y="y",
            tooltip=["target"],
            color=alt.Color("target:N"),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.01))
        )
        .add_params(selection).properties(
        title={
            "text": "2 Dimension Embedding Scatter Plot of 4 animal classes",
            "subtitle": ["4 Image Classes"]
        }, width=800
        )
    )   

    # mo.vstack([
    #     mo.md("Static 2 Dimension Embedding Scatter Plot of 4 animal classes"),
    #     chart
    # ])
    chart
    return (chart,)


@app.cell
def _(chart):
    mo_chart = mo.ui.altair_chart(chart)
    mo.vstack([
        mo.md("Interactive 2 Dimension Embedding Scatter Plot of 4 animal classes"),
        mo.md("Select images and image clusters, drag and zoom the chart"),
        mo_chart
    ])
    return (mo_chart,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For the chart above, use CTRL or COMMAND and zoom to zoom in an pan the chart.  You can also select the classes in the legend to highlight where some of the embeddings are not part of their cluster.
    """)
    return


@app.cell
def _():
    def summarize_class_column(df: pd.DataFrame) -> pd.DataFrame:
        summary_df = (
            df["target"]
            .value_counts(dropna=False)
            .rename_axis("target")
            .reset_index(name="count")
        )

        summary_df["percentage"] = (
            summary_df["count"] / summary_df["count"].sum() * 100
        ).round(2)

        return summary_df


    def create_class_summary_chart(summary_df):
        bars = (
            alt.Chart(summary_df)
            .mark_bar()
            .encode(
                x=alt.X("target:N", title="Target"),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color("target:N", legend=None),
                tooltip=["target", "count", "percentage"]
            )
        )

        text = (
            alt.Chart(summary_df)
            .mark_text(dy=-5)
            .encode(
                x="target:N",
                y="count:Q",
                text="count:Q"
            )
        )

        return (bars + text).properties(
            title="Target Distribution",
            width=500,
            height=300
        )

    return (summarize_class_column,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Using embeddings to verify image labeling

    The embedding clusters can be used to verify that the images are labeled correctly.  We can investigate each cluster and identify any images that are party of the minority class.  For example, if more images in a cluster are of one animal, we can look at any images that are labeled as part of another animal to confirm it is labeled correctly.
    """)
    return


@app.cell
def _(mo_chart, summarize_class_column):
    summary_df = summarize_class_column(mo_chart.value)
    summary_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Select each cluster to see what classes make up each cluster.

    You will see some interesting results.
    """)
    return


@app.cell
def _():
    target_name = mo.ui.text(value="e.g. cow", label="Name of target value to view in the selected cluster:")
    target_name
    return (target_name,)


@app.cell
def _(mo_chart, target_name):
    # get the index value of the selected target value
    # good way to check that the image and label are accurate and investigate why a label is mis-clustered
    target_name.value
    target_indexes = list(mo_chart.value.query('target==@target_name.value').index)
    return (target_indexes,)


@app.function
def get_filepath_filename_tuples(
    df: pd.DataFrame, indexes: list[int]
) -> list[tuple[str, str]]:
    results = []

    for idx in indexes:
        filepath = str(df.loc[idx, "filepath"])
        filename = Path(filepath).name
        results.append((filepath, filename))

    return results


@app.cell(hide_code=True)
def _(dropdown):
    mo.md(f"""
    #### Review Images for the specified target value of: **{dropdown.value}**
    """)
    return


@app.cell
def _(image_files_df, target_indexes):
    curious_image_embeddings = get_filepath_filename_tuples(image_files_df, target_indexes)
    # mo.vstack([mo.image(i, width=300, caption=d) for i,d in curious_image_embeddings ])

    mo.vstack(
        items=[
            mo.hstack(
                items=(
                    mo.image(i, width=300),
                    mo.md(f'<a href="{i}" target="_blank">{d}</a>'),
                    mo.ui.checkbox(label=f"Select image to remove: {'/'.join(i.split('/')[-4:])}")
                ),
                align="center",
                justify="start",
                gap=4
            )
            for i, d in curious_image_embeddings
        ],
        align="start"
    )
    return (curious_image_embeddings,)


@app.cell
def _(curious_image_embeddings):
    curious_image_embeddings[0:1]
    return


@app.cell
def _():
    mo.vstack(
        [
            mo.md("**Training Image Label Observations**"),

        mo.accordion(
            {
                "Train Spider Cluster": 
                mo.md("""
                * Notice that there is one 'cow' image in the Spider cluster.  However, when you look at that 'cow' image you see that it is actually 2 squirrels.  This is clearly the wrong image.
                """),
                "Train Horse Cluster": 
                mo.md("""
                * The horse cluster has images of cows, elephants and spiders.  Looking at the cow images, you see there are clearly some mislabeled images.  Some of the 'cow' images are actually horse images and should be reclassified.
                The elephant image is actually an image of a statue and should be removed.
                The spider image is actually an image of a car and should also be removed.
                """),
                "Train Cow Cluster": 
                mo.md("""
                * The cow cluster has 9 images of horses.  We can see that some of the images are not of horses and those should be removed.  
                """),
                "Train Elephant Cluster": 
                mo.md("""
                * The elephant cluster has images of cows and a spider. The cow images seem reasonably cows, but the spider image is an image of a monkey and should be removed.
                """),

            }
        )
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Create a Scikit-Learn Machine Learning Pipeline

    In this section we will create a scikit-learn machine learning pipeline to train a model on the Train image embeddings and then use the Valid image embeddings to validate the models performance.


    The first step is to create a Scikit-Learn Pipeline using Embetter classes to read the training images from the filesystem, and CLIP encode each image.  That pipeline looks like the following:


    ```python
    training_image_embedding_pipeline = make_pipeline(
           ColumnGrabber("filepath"),
          ImageLoader(convert="RGB"),
          ClipEncoder(),
        )
    ```

    We are going to read all of the file paths to the training images into a dataframe in a column called, 'filepath', and another column called, 'target' which will have the image label, e.g. 'cow', 'elephant', 'horse', 'spider'.

    The filepath will be sent to the ImageLoader which will load the image from the filesystem and then send that image to the ClipEncoder.  This will produce a 512 element vector for each image.

    To create a dataset suitable for Scikit-Learn

    ```python
    training_X = image_embedding_pipeline.fit_transform(training_files_df)
    training_y = training_files_df['target']
    ```

    We now have a dataset suitable to use with a Scikit-Learn LogisticRegression model to train a model.  This model will then be used to Validate the accuracy of the model performance and then used to create a Test submission.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### LogisticRegression Model Training

    This section will demonstrate how to go from Images to a trained LogisticRegression classification model.
    """)
    return


@app.cell
def _(create_filepaths_df):
    training_files_df = create_filepaths_df(dir_name='Train')
    training_files_df.shape
    return (training_files_df,)


@app.cell
def _():
    training_image_embedding_pipeline = make_pipeline(
           ColumnGrabber("filepath"),
          ImageLoader(convert="RGB"),
          ClipEncoder(),
        )
    return


@app.cell
def _(image_embedding_pipeline, training_files_df):
    # convert the filepaths to embeddings
    training_X = image_embedding_pipeline.fit_transform(training_files_df)
    training_y = training_files_df['target']
    print(training_X.shape)
    print(training_y.shape)
    return training_X, training_y


@app.cell
def _():
    # create a baseline, default model
    training_lr_model = LogisticRegression(solver='lbfgs', max_iter=1_000)
    return (training_lr_model,)


@app.cell
def _():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return (cv,)


@app.cell
def _(cv, training_X, training_lr_model, training_y):
    training_scores = cross_val_score(training_lr_model, training_X, training_y, cv=cv)
    print(training_scores)
    print(f"Accuracy: {training_scores.mean():.3f} (+/- {training_scores.std():.2f})")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now train the model on all of the Training images
    """)
    return


@app.cell
def _(training_X, training_y):
    # train the model on all of the data
    lr_model = LogisticRegression(solver='lbfgs', max_iter=1_000)
    lr_model.fit(training_X, training_y)
    return (lr_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### LogisticRegression Model Validation

    Using the Validation images, and the training LogisticRegression model.  See what the accuracy is on images the model has never seen before.
    """)
    return


@app.cell
def _(create_filepaths_df, lr_model):
    validation_files_df = create_filepaths_df(dir_name='Valid')

    # create pipeline to read the filepath column, load the image, and encode the image
    validation_image_embedding_pipeline = make_pipeline(
      ColumnGrabber("filepath"),
      ImageLoader(convert="RGB"),
      ClipEncoder(),
    )

    validation_X = validation_image_embedding_pipeline.fit_transform(validation_files_df)
    validation_y = validation_files_df['target']

    y_pred = lr_model.predict(validation_X)
    print(accuracy_score(validation_y, y_pred))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Even with some of the noisy training data, on Validation images that the model has never seen we obtained an accuracy of 99.4%.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Simulate a Kaggle Course Submission
    """)
    return


@app.cell
def _(create_filepaths_df):
    @mo.cache
    def calculate_test_image_embeddings(test_files_df:pd.DataFrame)-> np.ndarray:

        image_embedding_pipeline = make_pipeline(
           ColumnGrabber("filepath"),
           ImageLoader(convert="RGB"),
           ClipEncoder(),
        )

        test_X = image_embedding_pipeline.fit_transform(test_files_df)
        return test_X

    def run_model_on_test_images(model: LogisticRegression):
        print("Testing the model")
        test_files_df = create_filepaths_df(dir_name='Test', dirs=[""])

        test_X = calculate_test_image_embeddings(test_files_df)

        y_pred = model.predict(test_X)

        submission_df = create_submission_csv(test_files_df, y_pred)
        submission_df.to_csv('test_submission.csv', index=False)

    def create_submission_csv(df: pd.DataFrame, class_list: List[str]) -> pd.DataFrame:
        if 'filepath' not in df.columns:
            raise ValueError("DataFrame must contain a 'filepath' column.")
        if len(df) != len(class_list):
            raise ValueError("Length of class list must match number of rows in DataFrame.")

        result_df = pd.DataFrame()
        result_df['ID'] = df['filepath'].apply(lambda x: os.path.basename(x))
        result_df['CLASS'] = class_list
        return result_df


    return calculate_test_image_embeddings, run_model_on_test_images


@app.cell
def _():
    create_test_submission_btn = mo.ui.run_button(
        label="Click to create a test submission using the trained LogisticRegression model",
    )
    create_test_submission_btn
    return (create_test_submission_btn,)


@app.cell
def _(create_test_submission_btn, lr_model, run_model_on_test_images):
    mo.stop( not create_test_submission_btn.value)

    run_model_on_test_images(lr_model)
    return


@app.cell
def _():

    def compare_classifications(predicted_csv: str, ground_truth_csv: str) -> pd.DataFrame:
        # Read both CSV files
        predicted_df = pd.read_csv(predicted_csv)
        ground_truth_df = pd.read_csv(ground_truth_csv)

        # Rename the CLASS column in the second dataframe
        ground_truth_df = ground_truth_df.rename(columns={'CLASS': 'TRUE_CLASS'})

        # Merge on the ID column
        merged_df = pd.merge(predicted_df, ground_truth_df, on='ID')

        # Create CORRECT column: 1 if CLASS = TRUE_CLASS, else 0
        merged_df['CORRECT'] = (merged_df['CLASS'] == merged_df['TRUE_CLASS']).astype(int)

        return merged_df



    def evaluate_classification_results(submission_results_df: pd.DataFrame) -> float:
        if not {'CLASS', 'TRUE_CLASS', 'CORRECT'}.issubset(submission_results_df.columns):
            raise ValueError("DataFrame must contain 'CLASS', 'TRUE_CLASS', and 'CORRECT' columns.")

        # Accuracy
        accuracy = submission_results_df['CORRECT'].mean()
        print(f"****** Submission Accuracy: {accuracy:.2%}")

        # Confusion matrix
        print("\nConfusion Matrix:")
        confusion_matrix = pd.crosstab(
            submission_results_df['TRUE_CLASS'],
            submission_results_df['CLASS'],
            rownames=['Actual'],
            colnames=['Predicted'],
            dropna=False
        )
        print(confusion_matrix)
        return accuracy


    def grade_submission()->float:
        submission_df = compare_classifications(predicted_csv="test_submission.csv", ground_truth_csv="test_true_values.csv")
        submission_df.to_csv('submission_results.csv', index=False)

        return evaluate_classification_results(submission_df)


    return (
        compare_classifications,
        evaluate_classification_results,
        grade_submission,
    )


@app.cell
def _():
    grade_test_submission_btn = mo.ui.run_button(label="Grade Test Submission")
    grade_test_submission_btn
    return (grade_test_submission_btn,)


@app.cell
def _(grade_submission, grade_test_submission_btn):
    mo.stop( not grade_test_submission_btn.value )

    test_submission_accuracy = grade_submission()
    return (test_submission_accuracy,)


@app.cell(hide_code=True)
def _(test_submission_accuracy):
    mo.md(f"""
    The test submission accuracy using image embeddings [**{round(test_submission_accuracy,5)}**] exceeds the very best score using a custom Keras model [**0.97560**].  Using image embeddings with a Scikit-Learn model took significantly less time to develop and far less compute time to achieve superior results
    """)
    return


@app.function
def display_incorrect_predictions(df=None, dataset_path="dataset/Test/", submission_file_name="submission_results.csv"):
    """
    Displays incorrect predictions using marimo notebook elements.
    """
    if df is None:
        df = pd.read_csv(submission_file_name)

    incorrect_df = df[df['CORRECT'] == 0]

    if incorrect_df.empty:
        return mo.md("🎉 No incorrect predictions!")

    elements = []
    for idx, row in incorrect_df.iterrows():
        img_id = str(row['ID'])
        img_path = f"{dataset_path}{img_id}"
        predicted_class = row['CLASS']
        true_class = row['TRUE_CLASS']

        try:
            # Use marimo's image element
            img = mo.image(src=img_path, width=400)

            # Create a label for the prediction
            label = mo.md(f"**Predicted:** {predicted_class} | **True:** {true_class} [{img_id}]")

            # Combine image and label in a vstack
            elements.append(mo.vstack([img, label], align='center'))


        except Exception as e:
            elements.append(mo.md(f"⚠️ Could not display image {img_path}: {e}"))

    return mo.vstack(elements)


@app.cell
def _():
    display_incorrect_predictions_btn = mo.ui.run_button(label = "Display incorrect test submission predictions")
    display_incorrect_predictions_btn
    return (display_incorrect_predictions_btn,)


@app.cell
def _(display_incorrect_predictions_btn):
    mo.stop(not display_incorrect_predictions_btn.value )

    display_incorrect_predictions()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Zero Shot Image Classification
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Zero-shot image classification, using embeddings, is a technique for classifying images without training a model specifically on those image targets.

    This technique relies on the semantic context captured in the image embeddings as well as text embeddings.

    **Embeddings** are just numerical representations that capture the meaning/context of either an image or text.  Models like CLIP map both images and text into the same vector space, so that similar concepts are 'close' together.

    **Shared image-text embedding space** allows for models like CLIP, that have been trained on image/text pairs to understand the relationship between visual features and the language and text that describes the visual features.

    **Zero-shot classification process** involves converting images to embeddings, and then converting the labels or text description into an embedding, the and then compare the similarity of the text embeddings with the images.  The text embedding most similar to the image embedding will be considered the predicted target/label value.

    The reason it is called **zero-shot** is because we never explicity train the images and specific labels with a model like we did in the previousl section with Scikit-Learn.  This technique also means that an image can have multiple labels.  For example, if we are creating a classifier for cats and dogs, while the label for training might be 'cat', we will see that 'feline' is also considered a valid label for cat images.

    The limitation of Zero-shot image classification is that the performance may be lower than a purpose built trained supervised model.

    In this section will use Zero-shot Image classification to see how this technique will work with the 4 animal classification.  Then we will extend this 4 animal classification to include cats and dogs, and with not training see how the technique handles completely new image labels.
    """)
    return


@app.cell
def _(chart):
    chart
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Using the target names, create word embeddings for each of the targets and map the target word embeddings onto the image embeddings.  Our goal is to see if the embeddings can capture semantic meaning from the image and the text and associate those meanings.  If embeddings can do this, then we would expect the 'word' embedding to be co-located with the image embeddings.
    """)
    return


@app.cell
def _():
    target_input_values = mo.ui.text(value="cow,elephant,horse,spider", label="Target Names: ", full_width=True)
    target_input_values
    return (target_input_values,)


@app.cell
def _(target_input_values):
    # bovine
    # pachyderm
    # equine
    # arachnid
    # more than four legs
    # very large ears

    target_names = target_input_values.value.split(",")
    return (target_names,)


@app.cell
def _(image_clip_encoder, target_names):
    target_embeddings = image_clip_encoder.transform(target_names)
    return (target_embeddings,)


@app.cell
def _(reducer, target_embeddings):
    reduced_word_embeddings = reducer.transform(target_embeddings)
    return (reduced_word_embeddings,)


@app.cell
def _(reduced_word_embeddings):
    reduced_word_embeddings.shape
    return


@app.cell
def _(reduced_word_embeddings):
    reduced_word_embeddings
    return


@app.cell
def _(reduced_word_embeddings, target_names):
    # create a chart layer for the word embeddings to overlay on the image embedding chart
    annotation_df = pd.DataFrame(reduced_word_embeddings, columns=["x", "y"])
    annotation_df["label"] = target_names

    annotation_layer = (
        alt.Chart(annotation_df)
        .mark_point(shape="cross", size=700, color="cyan", filled=True, opacity=1)
        .encode(
            x="x:Q",
            y="y:Q",
            tooltip=["label:N"]
        )
    )

    text_layer = (
        alt.Chart(annotation_df)
        .mark_text(dx=30, dy=-15, fontSize=26, fontStyle="bold")
        .encode(
            x="x:Q",
            y="y:Q",
            text="label:N"
        )
    )
    return annotation_layer, text_layer


@app.cell
def _():
    return


@app.cell
def _(annotation_layer, chart, text_layer):

    image_text_chart = alt.layer(chart + annotation_layer + text_layer )

    mo.vstack([mo.md("### Image and Text Embeddings"), image_text_chart])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Because we are mapping text embeddings onto the image image embeddings, we are not constrained to just those target values we are interested in predicted.  We can try any word or phrase.  Go back to the Target Names input and try some of the following:


    * bovine  <- cow
    * pachyderm  <- elephant
    * equine  <- horse
    * arachnid  <- spider
    * more than four legs  <- spiders are on the only animal in this collection with more than 4 legs
    * a trunk for a nose <- elephants have a trunk nose

    Notice besides words we can use a description, and the embedding will map the semantic context of the phrase to the image that best represents that phrase
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Non-Target Words

    What happens if we try to use a non-target word?  How is a word that has nothing to do with the pictures get mapped?

    Try any word you like, but for this notebook lets try:

    * YoYo
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Why was the word **YoYo** mapped to Spider?

    If we could visualize 512 dimensional space, we might see the word **YoYo** is set off and apart from any of the animals but might reside 'closer' to spider than the other animals.  Then during the reduction of dimensionality from 512 to 2, information is lost and those distances become distorted.

    So you cannot use this approach to determine if something is **NOT** one of the 4 animals.

    **NOTE**: Before going to the next cell, make sure to reset the target values to cow,elephant,horse,spider
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Create a Zero-Shot Classifier

    Because we do not need to train a model and validate a model, we can go right to the test submission images and create a test submission to see how the Zero-Shot Classifier performs.  This is one of the benefits of using embeddings and zero shot techniques.
    """)
    return


@app.cell
def _(compare_classifications, evaluate_classification_results):
    def softmax_from_cosine_similarities(similarities):
        """
        Converts a list of cosine similarities to softmax probabilities.

        Args:
            similarities (list or np.ndarray): A list or array of cosine similarity scores.

        Returns:
            np.ndarray: Softmax probabilities.
        """
        similarities = np.array(similarities)
        # Subtract max for numerical stability
        exp_values = np.exp(similarities - np.max(similarities))
        softmax_probs = exp_values / np.sum(exp_values)
        return softmax_probs

    def cosine_similarity2(u: np.ndarray, v: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    def create_submission_df(df: pd.DataFrame, class_list: List[str]) -> pd.DataFrame:
        if 'filepath' not in df.columns:
            raise ValueError("DataFrame must contain a 'filepath' column.")
        if len(df) != len(class_list):
            raise ValueError("Length of class list must match number of rows in DataFrame.")

        result_df = pd.DataFrame()
        result_df['ID'] = df['filepath'].apply(lambda x: os.path.basename(x))
        result_df['CLASS'] = class_list
        return result_df

    def grade_zero_shot_submission():
        submission_df = compare_classifications(predicted_csv="zero_shot_test_submission.csv", ground_truth_csv="test_true_values.csv")
        submission_df.to_csv('zero_shot_submission_results.csv', index=False)

        evaluate_classification_results(submission_df)


    return create_submission_df, grade_zero_shot_submission


@app.cell
def _(
    calculate_test_image_embeddings,
    create_filepaths_df,
    create_submission_df,
    target_embeddings,
    target_names,
):
    def create_zero_shot_test_submission():
        print("Test Zero Shot model")
        test_files_df = create_filepaths_df(dir_name='Test', dirs=[""])

        test_X = calculate_test_image_embeddings(test_files_df)

        y_preds = []
        for i, row in enumerate(test_X):
            # for each test image embedding
            image_embedding = test_X[i]
            most_similar_target = None
            most_similar_similarity = -1
            for x,target_name in enumerate(target_names):
                # check each possible target name embedding to find most similar
                target_embedding = target_embeddings[x]
                similarity = cosine_similarity([image_embedding], [target_embedding])
                if similarity > most_similar_similarity:
                    most_similar_similarity = similarity
                    most_similar_target = target_name
            y_preds.append(most_similar_target)

        submission_df = create_submission_df(test_files_df, y_preds)
        submission_df.to_csv('zero_shot_test_submission.csv', index=False)


    return (create_zero_shot_test_submission,)


@app.cell
def _():
    create_zero_shot_submission_btn = mo.ui.run_button(label="Create Test Submission")
    create_zero_shot_submission_btn
    return (create_zero_shot_submission_btn,)


@app.cell
def _(create_zero_shot_submission_btn, create_zero_shot_test_submission):
    mo.stop(not create_zero_shot_submission_btn.value )

    create_zero_shot_test_submission()
    return


@app.cell
def _():
    grade_zero_shot_submission_btn = mo.ui.run_button(label="Grade Test Submission")
    grade_zero_shot_submission_btn
    return (grade_zero_shot_submission_btn,)


@app.cell
def _(grade_zero_shot_submission, grade_zero_shot_submission_btn):
    mo.stop(not grade_zero_shot_submission_btn.value )

    grade_zero_shot_submission()
    return


@app.cell
def _():
    display_incorrect_zero_shot_btn = mo.ui.run_button(label="Display Incorrect Zero Shot Predictions")
    display_incorrect_zero_shot_btn
    return (display_incorrect_zero_shot_btn,)


@app.cell
def _(display_incorrect_zero_shot_btn):
    mo.stop(not display_incorrect_zero_shot_btn.value)
    display_incorrect_predictions(submission_file_name="zero_shot_submission_results.csv")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Vision Language Model Classifier

    In this section we will use a Vision Language Model ( VLM ) to classify images.  This section will use the **qwen3-vl** and a locally running Ollama.


    ```python
    prompt_text = "\"\"What animal is in this image?
    You must pick from cow, horse, spider, elephant and you can only return that single animal name.
    Even if the picture shows a different animal, you have to pick the best match."\"\"
    ```
    """)
    return


@app.cell
def _():
    # Text prompt to send with the image
    prompt_text = "What animal is in this image?  You must pick from cow, horse, spider, elephant and you can only return that single animal name.  Even if the picture shows a different animal, you have to pick the best match."
    return (prompt_text,)


@app.cell
def _():

    # Define the model and default image path
    MODEL_NAME = 'qwen3-vl'
    DEFAULT_IMAGE_PATH = './spider/1010975.jpg' # Use the correct relative path based on the project structure

    def get_animal_from_image(image_path, prompt_text):
        """
        Sends an image to the LLM and returns the response content.
        :param image_path: Path to the image file
        :return: Response message content from the model
        """
        # Check if the image file exists
        if not os.path.exists(image_path):
            return f"Error: Image file '{image_path}' does not exist."


        print(f"Sending request to {MODEL_NAME} with image: {image_path}")

        try:
            # Send the request with both text content and the image
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt_text,
                        'images': [image_path] # Pass the local file path
                    }
                ],
            )
            return response['message']['content'] # or access using dot notation: return response.message.content

        except Exception as e:
            return f"An error occurred: {e}\nPlease ensure your local Ollama server is running."




    return (get_animal_from_image,)


@app.cell
def _():
    start_vlm_btn = mo.ui.run_button(label="Start sending images to QWEN3 VLM")
    start_vlm_btn
    return (start_vlm_btn,)


@app.cell
def _(get_animal_from_image, prompt_text, start_vlm_btn, write):
    mo.stop(not start_vlm_btn.value )

    if not Path("./vlm_test_submission.csv").exists:
        # Default to a directory if not provided as an argument
        directory_to_process = './dataset/Test'

        results = {}

        if not os.path.isdir(directory_to_process):
            print(f"Error: Directory '{directory_to_process}' does not exist.")
        else:
            print(f"Scanning directory: {directory_to_process}")
            # Loop through all files in the directory
            print("Start sending images to VLM")
            for filename in sorted(os.listdir(directory_to_process)):
                file_path = os.path.join(directory_to_process, filename)
                # Only process files (skip directories)
                if os.path.isfile(file_path):
                    # Basic check for image extensions
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        result = get_animal_from_image(file_path, prompt_text)
                        results[filename] = result

            print("\Completed sending all images to VLM")
            # print(results)
            # Create the submission.csv file
            csv_file = 'vlm_test_submission.csv'
            print(f"\nWriting results to {csv_file}...")
            try:
                with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    write.writerow("ID,CLASS")
                    # Not explicitly requested a header, but typically CSVs might have one.
                    # However, the example "filename.jpg,prediction name" suggests no header or a simple format.
                    # The description says "a row for each filename and prediction".
                    for image_filename, vlm_prediction in results.items():
                        writer.writerow([image_filename, vlm_prediction])
                print(f"Successfully created {csv_file}")
            except Exception as e:
                print(f"Error writing to CSV: {e}")

    else:
        print("vlm_test_submission.csv already exists.  Skipping to avoid lengthy cell execution")
    return


@app.cell
def _():


    return


@app.cell
def _(compare_classifications, evaluate_classification_results):
    def grade_vlm_submission():
        submission_df = compare_classifications(predicted_csv="vlm_test_submission.csv", ground_truth_csv="test_true_values.csv")
        submission_df.to_csv('vlm_submission_results.csv', index=False)

        evaluate_classification_results(submission_df)

    return (grade_vlm_submission,)


@app.cell
def _():
    grade_vlm_btn = mo.ui.run_button(label="Grade VLM Results")
    grade_vlm_btn
    return (grade_vlm_btn,)


@app.cell
def _(grade_vlm_btn, grade_vlm_submission):
    mo.stop(not grade_vlm_btn.value)

    grade_vlm_submission()
    return


@app.cell
def _():
    display_incorrect_vlm_btn = mo.ui.run_button(label="Display Incorrect VLM Results")
    display_incorrect_vlm_btn
    return (display_incorrect_vlm_btn,)


@app.cell
def _(display_incorrect_vlm_btn):
    mo.stop(not display_incorrect_vlm_btn.value)
    display_incorrect_predictions(submission_file_name="vlm_submission_results.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
