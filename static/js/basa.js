

$(document).ready(function () {
const select = document.querySelector('#zalupa').getElementsByTagName('option');
var urlsub = window.location.href.split('?')[1];
for (let i = 0; i < select.length; i++) {
    if (select[i].value === urlsub) select[i].selected = true;
}
});
$(document).ready(function () {
  $("#form").submit(function (event) {
    var zalupa = $('#zalupa').val();
    var url = window.location.href.split('?')[0];
    var url = url + '?' + zalupa;
    $.ajax({
      type: "POST",
      url: "/get_table",
      data: {
        zalupa:zalupa,
      },
      success: function () {
        location.assign(url);

      }
    });
    return false; //<---- move it here
  });}
);


$(document).ready(function () {

  $("#form_input").submit(function (event) {
    var url = location.href;
    var update_table = url.substring(url.indexOf("?")+1);
    var datas = []
    $('#form_input input, #formId select').each(
    function(index){
        var input = $(this);
        datas.push(input.val())

    }
);
    var users_to_delete = $('#form_input').val();
    var datas = datas.toString();
    $.ajax({
      type: "POST",
      url: "/get_input",
      data: {
        datas: datas,
        update_table:update_table,
      },

      success: function () {
        location.reload();
      }
    });
    return false; //<---- move it here
  });}
);



$(document).ready(function () {
  $("#form_delete").submit(function (event) {
    var users_to_delete = $('#users_to_delete').val();
    var url = location.href;
    var update_table = url.substring(url.indexOf("?")+1);
    $.ajax({
      type: "POST",
      url: "/delete_user",
      data: {
        users_to_delete:users_to_delete,
        update_table:update_table,
      },

      success: function () {
        location.reload();
      }
    });
    return false; //<---- move it here
  });}
);