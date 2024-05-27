from flask import Flask, render_template
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # Load the Iris dataset
    iris = pd.read_csv('iris.csv', delimiter=',')
    
    # Generate plots
    plots = []
    
    # Plot 1: Sample of 10 rows
    plots.append(iris.sample(10).to_html(classes="table table-striped"))
    
    # Plot 2: Dataset shape
    shape_df = pd.DataFrame(iris.shape, columns=['Rows', 'Columns'], index=[''])
    plots.append('<h3>Dataset Shape</h3>' + shape_df.to_html(classes="table table-striped"))
    
    # Plot 3: Dataset info
    info_df = iris.info().to_frame()
    plots.append('<h3>Dataset Info</h3>' + info_df.to_html(classes="table table-striped"))
    
    # Plot 4: Group by Species
    species_count_df = iris.groupby('Species').size().to_frame(name='Count')
    plots.append('<h3>Group by Species</h3>' + species_count_df.to_html(classes="table table-striped"))
    
    # Plot 5: Descriptive statistics
    plots.append('<h3>Descriptive Statistics</h3>' + iris.describe().to_html(classes="table table-striped"))
    
    # Plot 6: Null values
    null_values_df = iris.isnull().sum().to_frame(name='Null Values')
    plots.append('<h3>Null Values</h3>' + null_values_df.to_html(classes="table table-striped"))
    
    # Plot 7: KDE plots
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=iris, x='SepalLengthCm', y='SepalWidthCm', hue="Species", fill=True)
    plt.title('KDE Plot: Sepal Length vs. Sepal Width')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plots.append('<img src="data:image/png;base64,{}">'.format(base64.b64encode(img.getvalue()).decode()))
    plt.close()
    
    # Render the HTML template with the plots
    return render_template('index.html', plots=plots)

if __name__ == '__main__':
    app.run(debug=True)
