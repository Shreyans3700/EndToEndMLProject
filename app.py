from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/predictData", methods=["GET", "POST"])
def predict():
    # If someone directly hits /predictData → show form instead of error
    if request.method == "GET":
        return render_template("home.html")

    # POST → actual prediction logic
    data = CustomData(
        gender=request.form.get("gender"),
        race_ethinicity=request.form.get("ethnicity"),
        parental_level_of_education=request.form.get("parental_level_of_education"),
        lunch=request.form.get("lunch"),
        test_preparation_course=request.form.get("test_preparation_course"),
        reading_score=float(request.form.get("reading_score")),
        writing_score=float(request.form.get("writing_score")),
    )

    request_df = data.get_data_as_frame()

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(features=request_df)

    return render_template("home.html", results=round(results[0], 2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
