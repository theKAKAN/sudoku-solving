from flask import Flask, render_template, request, jsonify

from generator import Solver

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# TODO: Implement SSE, capture stdout and pipe it to the page

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save('./output/uploads/'+ file.filename)
    solve = Solver('./output/uploads/'+ file.filename)
    solve.preprocessImage()
    solve.findContours()
    return render_template('image.html',
        sudoku_grid = solve.genCells(),
        solved_puzzle = solve.solve()
    )

if __name__ == '__main__':
    app.run(debug=True)
