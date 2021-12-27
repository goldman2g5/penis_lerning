

import pymysql
from flask import Flask, render_template, request, redirect
from pymysql.cursors import DictCursor

app = Flask(__name__)

dbh = pymysql.connect(
    host='185.12.94.106',
    user='2p1s23',
    password='941-961-748',
    db='2p1s23',
    charset='utf8mb4',
    cursorclass=DictCursor,
    autocommit=True
)


@app.route('/', methods=['post', 'get'])
def main():
    return render_template('glavnaya.html')


@app.route('/registration', methods=['post', 'get'])
def registration():

    message = 'Для регистрации заполните форму ниже'
    login, password, username = request.form.get('login'), request.form.get('password'), request.form.get(
        'username')  # запрос к данным формы

    if login and password and username and len(login) >= 4 and len(password) >= 4 and len(username) >= 4:
        cur = dbh.cursor()
        cur.execute(
            f'SELECT login, password, name FROM users WHERE login = "{login}" OR password = "{password}" OR name = "{username}";')
        huy = cur.fetchone()

        if huy is None:
            cur.execute(
                f'INSERT INTO users (id, login, password, name) VALUES (NULL, "{login}", "{password}", "{username}");')
            return redirect("/login")

        else:
            message = "Логин, пароль или имя уже используются, попробуйте другие"

    return render_template('reg.html', message=message)


@app.route('/login', methods=['post', 'get'])
def login():
    message = 'Для входа введите логин и пароль'
    login, password = request.form.get('username'), request.form.get('password')  # запрос к данным формы

    if login is not None or password is not None:
        cur = dbh.cursor()
        cur.execute(f'SELECT login, password FROM users WHERE login = "{login}" AND password = "{password}";')
        a = cur.fetchone()

        if a and a['login'] == login and a['password'] == password:
            return redirect("/neyronka")
        else:
            message = "Wrong username or password"
    return render_template('login.html', message=message)

@app.route('/neyronka', methods=['post', 'get'])
def neyronka():
    return render_template('neyronka.html')


@app.route('/garfield_race', methods=['post', 'get'])
def gonki():
    return render_template('gonki.html')

column = None
table = None
input_ = None
var = {1, 2}
@app.route('/get_table', methods=['GET', 'POST'])
def get_name():
    global var
    table = request.form['zalupa']
    if table:
        cur = dbh.cursor()
        cur.execute(f"SELECT *FROM {table}")
        var = cur.fetchall()

    return render_template('baza.html')

@app.route('/get_input', methods=['GET', 'POST'])
def get_input():
    global new_name
    new_name = request.form['name']
    update_table = request.form['update_table']
    print(update_table)
    cur = dbh.cursor()
    cur.execute(f"SELECT *FROM {update_table}")
    ids = cur.fetchall()
    new_id = []
    for i in ids:
        z = i.get('id')
        z = int(z)
        new_id.append(z)
    new_id = new_id[-1]
    new_id = new_id + 1

    cur.execute(f'INSERT INTO {update_table} VALUES ("{new_id}", "{new_name}");')

    return render_template('baza.html')

@app.route('/delete_user', methods=['GET', 'POST'])
def delete_user():
    global new_name
    users_to_delete = request.form['users_to_delete']
    update_table = request.form['update_table']
    cur = dbh.cursor()
    cur.execute(f'SET FOREIGN_KEY_CHECKS = 0;')
    cur.execute(f'DELETE FROM {update_table} where {update_table}. id = {users_to_delete}')
    print("Record deleted successfully")

    return render_template('baza.html')

@app.route('/gigabaza', methods=['post', 'get'])
def baza():

    return render_template('baza.html', var=var)


if __name__ == "__main__":
    app.run(debug=True)

# cur.execute(f"SELECT admins.id, admins.name, room.id, room.room_name FROM admins, room, room_rent, console, guests")
# room.console_id, room.admin_id, room_rent.id, room_rent.rent_time, room_rent.guest_id, room_rent.room_id
