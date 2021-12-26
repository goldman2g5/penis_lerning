

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


keys_list = []
value_list = []
@app.route('/get_piskka', methods=['GET', 'POST'])
def get_name():
    name = request.form['zalupa']
    if name:
        cur = dbh.cursor()
        cur.execute(f"SELECT *FROM {name}")
        var = cur.fetchone()
        print(var)
        global keys_list
        global value_list
        global c

        value_list = list(var.values())
        keys_list = list(var.keys())
        c = 0

    return render_template('baza.html')


@app.route('/gigabaza', methods=['post', 'get'])
def baza():
    var1 = keys_list[0] if len(keys_list) >= 1 else ""
    var2 = keys_list[1] if len(keys_list) >= 2 else ""
    var3 = keys_list[2] if len(keys_list) >= 3 else ""
    var4 = keys_list[3] if len(keys_list) >= 4 else ""
    var5 = keys_list[4] if len(keys_list) >= 5 else ""
    val1 = value_list[0] if len(value_list) >= 1 else ""
    val2 = value_list[1] if len(value_list) >= 2 else ""
    val3 = value_list[2] if len(value_list) >= 3 else ""
    val4 = value_list[3] if len(value_list) >= 4 else ""
    val5 = value_list[4] if len(value_list) >= 5 else ""

    return render_template('baza.html', keys_list=keys_list, value_list=value_list, list_len=list(range(len(value_list))))


if __name__ == "__main__":
    app.run(debug=True)

# cur.execute(f"SELECT admins.id, admins.name, room.id, room.room_name FROM admins, room, room_rent, console, guests")
# room.console_id, room.admin_id, room_rent.id, room_rent.rent_time, room_rent.guest_id, room_rent.room_id
