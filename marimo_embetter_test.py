# /// script
# requires-python = ">=3.11"
# dependencies = [
#         "embetter[sbert]>=0.7.0",
#        "marimo[recommended]>=0.20.4",
#        "pillow>=12.1.1",
#        "scikit-learn>=1.8.0",
#        "umap-learn>=0.5.11",
# ]
# ///
import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    from embetter.multi import ClipEncoder


    return (ClipEncoder,)


@app.cell
def _(ClipEncoder):
    ClipEncoder()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
