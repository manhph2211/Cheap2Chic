function createAudioHTML(path) {
    return '<audio controls controlslist="nodownload" class="px-1"> <source src=' +
        path +
        ' type="audio/wav">Your browser does not support the audio element.</audio>';
  }
  
  
  function generateExampleRow(table_row, base_path, paddedNumber, folders, col_offset) {
    for (var i = 0; i < folders.length; i++) {
        if (folders[i] == 'devices') {
            let p = base_path + folders[i] + '/' + paddedNumber + '.txt';
            let cell = table_row.cells[col_offset + i];
            var req = new XMLHttpRequest();
            req.open("GET", p, false);
            req.send(null);
            cell.innerHTML = '<font size="-1">' + req.responseText + '</font>';
        } else {
            let p = base_path + folders[i] + '/' + paddedNumber + '.wav';
            let cell = table_row.cells[col_offset + i];
            cell.innerHTML = createAudioHTML(p); 
        }
    }
  }
  
  
  function generateT2A(tableId, e) {
    let table = document.getElementById(tableId);
    let folders = ['devices', 'input', 'target', 'output'];
  
    for (var i = 0; i < 6; i++) {
      let paddedNumber = i.toString().padStart(5, '0');  
      generateExampleRow(table.rows[1 + i], 'assets/'+e+'/', paddedNumber, folders, 0);
    }
  }

  $(document).ready(function() {
    generateT2A('supervision-efficiency-table', 1);
  });

  for (let e = 1; e <= 5; e++) {
    let id = '#supervision-efficiency-table-' + e;
    $(id).click(function() {
      generateT2A('supervision-efficiency-table', e);
      $(id).parent().siblings().removeClass('active');
      $(id).parent().addClass('active');
      return false;
    });
  }