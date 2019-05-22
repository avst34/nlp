var handlebars = require('handlebars');
var fs = require('fs');
var template = fs.readFileSync('./ch_template.handlebars.txt', 'utf8');
// var context = JSON.parse(fs.readFileSync('/cs/labs/oabend/aviramstern/best_results/template_input.json', 'utf8'));
var context = JSON.parse(fs.readFileSync('/Projects/nlp/docs/chinese_results.json', 'utf8'));

var compiled = handlebars.compile(template);
var rendered = compiled(context);

console.log(rendered);
