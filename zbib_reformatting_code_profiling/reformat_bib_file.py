import re


def reorder_bibtex_article(entry):
    # Define the desired order of fields
    ordered_fields = [
        'author',
        'year',
        'title',  # Keep title in its original position.
        'journal',
        'volume',
        'issn',
        'number',
        'pages',
        'url',
        'doi'
    ]

    # Extract the identifier of the entry
    identifier = re.search(r'@article{([^,]+),', entry)
    if identifier:
        identifier = identifier.group(1).strip()
    else:
        return entry  # If not a valid entry, return as is

    # Extract fields
    fields = {}
    for field in ordered_fields:
        match = re.search(rf'{field}\s*=\s*{{(.*?)}}', entry, re.DOTALL)
        if match:
            # Preserve the original title formatting
            if field == 'title':
                fields[field] = match.group(0) + ','  # Add a comma to the extracted field
            else:
                fields[field] = match.group(0) + ','

    # Create the new entry in the desired order
    new_entry = f"@article{{{identifier},\n"
    for field in ordered_fields:
        if field in fields:
            new_entry += f"    {fields[field]}\n"
    new_entry = new_entry.rstrip('\n') + "\n}"  # Add closing brace without a comma

    return new_entry


def format_bib_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()

    # Split the content into individual entries
    entries = re.split(r'(?=@)', content.strip())  # Split on '@' but retain it in result
    formatted_entries = []

    for entry in entries:
        entry = entry.strip()
        if entry and entry.startswith('@article'):
            # Process only @article entries
            formatted_entries.append(reorder_bibtex_article(entry) + '\n\n')  # Add a blank line after each entry
        elif entry:  # Include non-article entries unchanged but no extra line
            formatted_entries.append(entry.strip() + '\n')  # Ensure no extra blank lines for non-article entries

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(''.join(formatted_entries))


# Specify the input and output files
input_bib_file = 'Msc_thesis.bib'
output_bib_file = 'Msc_thesis_cleaned.bib'

# Format the bibliography file
format_bib_file(input_bib_file, output_bib_file)

print("BibTeX entries have been reordered and saved to:", output_bib_file)