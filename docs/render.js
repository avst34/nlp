var handlebars = require('handlebars');
var fs = require('fs');
var template = fs.readFileSync('./template.handlebars.txt', 'utf8');
var context = JSON.parse(fs.readFileSync('./context.json', 'utf8'));

var compiled = handlebars.compile(template);
var rendered = compiled(context);

console.log(rendered);
