from flask import Flask, render_template, request

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URl"] = "sqlite:////debil.db"


@app.route('/login/', methods=['post', 'get'])
def login():
    message = ''
    username = request.form.get('username')  # запрос к данным формы
    password = request.form.get('password')

    if username == 'root' and password == 'pass':
        message = "Correct username and password"
    else:
        message = "Wrong username or password"
    return render_template('index.html', message=message)


@app.route('/user/<string:name>/<int:id>')
def user(name, id):
    return "User page: " + name + " - " + str(id)


if __name__ == "__main__":
    app.run(debug=True)

