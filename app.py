from src.logger import logging
from src.pipelines.prediction_pipeline import CustomData,PredictionPipeline
from flask import Flask,render_template,request,jsonify


app=Flask(__name__)
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            NoOfBedrooms=float(request.form.get('NoOfBedrooms')),
            NoOfBathrooms=float(request.form.get('NoOfBathrooms')),
            NoOfFloors=float(request.form.get('NoOfFloors')),
            FlatArea=float(request.form.get('FlatArea')),
            LotArea=float(request.form.get('LotArea')),
            BasementArea=float(request.form.get('BasementArea')),
            AreaOfTheHouseFromBasement=float(request.form.get('AreaOfTheHouseFromBasement')),
            LivingAreaAfterRenovation=float(request.form.get('LivingAreaAfterRenovation')),
            LotAreaAfterRenovation=float(request.form.get('LotAreaAfterRenovation')),
            AgeOfHouse=float(request.form.get('AgeOfHouse')),
            ConditionOfTheHouse=request.form.get('ConditionOfTheHouse'),
            OverallGrade=float(request.form.get('OverallGrade'))   
        )
        logging.info(data)
        final_new_data=data.get_data_as_dataframe()
        logging.info(final_new_data)
        predict_pipeline=PredictionPipeline()
        pred=predict_pipeline.predict(final_new_data)
        results=round(pred[0])
        return render_template('result.html',final_result=results)

    
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)