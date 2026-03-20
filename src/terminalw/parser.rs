pub use super::internal::parser::{Parser, ParserOption, disable_parser_number};

pub fn new_parser(options: &[ParserOption]) -> Parser {
    Parser::new_with_options(options)
}
