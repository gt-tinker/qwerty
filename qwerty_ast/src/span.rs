#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSpan {
    pub file: String, // Path or file name
    pub start: usize, // Start byte offset in the file
    pub end: usize,   // End byte offset (non-inclusive)
    pub line: usize,  // Line number (starting from 1)
    pub col: usize,   // Column number (starting from 1)
}

// eprintln!("{}:{}:{}: error: ...", span.file, span.line, span.col);
