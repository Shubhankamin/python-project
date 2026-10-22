INDIAN ROAD ACCIDENT SEVERITY CLASSIFICATION — PROJECT PACKAGE

Files:
- Indian_Road_Accident_Phase2_LabStyle.ipynb : simple notebook in lab-manual style
- indian_roads_dataset.xlsx : input dataset
- model_comparison_results.csv : actual model comparison table from a run
- Phase2_Final_Report.docx : report draft with actual observed metrics
- Phase2_Project_Presentation.pptx : presentation slides
- Viva_Questions_and_Answers.docx : viva preparation
- app.py : optional Streamlit demonstration
- requirements.txt : dependencies

Run notebook:
1. Open the .ipynb in Google Colab.
2. Upload indian_roads_dataset.xlsx when asked, or keep it in the same folder.
3. Run all cells and save the executed notebook with outputs.

Run optional demo locally:
1. Install requirements: pip install -r requirements.txt
2. Run: streamlit run app.py

Important:
- The model metrics are based on a stratified 80/20 split with random_state=42.
- Decision Tree had the highest macro F1 among the six model runs recorded here.
- Gradient Boosting had the highest accuracy but zero fatal-class recall in the recorded run.
- Re-run the notebook and confirm results before submission, because package versions and execution details can affect results.
- Dataset provenance is undocumented and the data may be synthetic. Do not claim real-world road safety validity.
