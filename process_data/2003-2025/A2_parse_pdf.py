import re
import os
import pdfplumber
import pandas as pd

out_folder = "data/prd/WHOCC/"
pdf_folder = "data/raw/WHOCC/"
columns = ["Test Virus","Reference Virus", "Titre"]

filename = '2019-feb.pdf'

CITY_ABBREVIATION_DICT = {
    "Chch": "Christchurch",
    "ND": "North Dakota",
    "HK": "Hong Kong",
    "NY": "New York",
    "BW": "Baden-Wurttemburg",
    "NH": "New Hampshire",
    "Jbg": "Johannesburg",
    "Jhb": "Johannesburg",
    "JHB": "Johannesburg",
    "Fin": "Finland",
    "SAust": "South Australia"
    
}

if filename in  ['2003-aug.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and "Table" in text and "Antigenic analyses of influenza A H3N2 viruses" in text:

                text = re.sub(r'\n(\d+)\n', r' \1\n', text)
                text = text.replace("\n|", " *")
                text = text.replace("<", "<40")
                lines = text.split('\n')

                for line in lines:
                    if line.startswith("Viruses Isolation"):
                        header_line = line
                        break

                virus_pattern = re.compile(r'A/[A-Za-z\-]+/[A-Za-z0-9\-]+/\d{2}')
                full_virus_names = []

                for line in lines:
                    matches = virus_pattern.findall(line)
                    for m in matches:
                        if m not in full_virus_names:
                            full_virus_names.append(m)


                tokens = header_line.split()[2:]
                col_names = []

                for t in tokens:
                    matches = [full for full in full_virus_names if full.startswith(t)]

                    if matches:
                        col_names.append(matches[0])
                    else:
                        cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                        if cityname:
                            matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                            col_names.append(matches[0])

                col_num = len(col_names)


                data = []
                for line in lines:
                    virus_match = virus_pattern.search(line)
                    if virus_match:
                        virus_name = virus_match.group()

                        elem = line.split(" ")

                        if virus_name not in col_names and len(elem) < 8:
                            continue

                        elem = elem[-col_num:]
                        data.append([virus_name] + elem)

                df = pd.DataFrame(data)
                df.columns = ['Test Virus'] + col_names

                long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                print(long_df)
                virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in ['2003-dec.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[11]
        text = page.extract_text()

        text = re.sub(r'\n(\d+)\n', r' \1\n', text)
        text = text.replace("\n|", " *")
        text = re.sub(r'\n(?![AV])', ' ', text)
        text = text.replace("|", "*")
        text = text.replace("<", "<40")
        text = text.replace("Haemagglutination inhibition titre1 Post infection ferret sera ", "")
        text = text.replace(" 1 . <40,<4040", "")
        lines = text.split('\n')

        virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')


        col_names = ['A/Panama/2007/99', 'A/New York/55/01', 'A/Hong Kong/1550/02', 'A/Egypt/130/02', 'A/Fujian/411/02', 'A/Sendai/4952/02', 'A/Finland/170/03']

        col_num = len(col_names)


        data = []
        for line in lines:
            virus_match = virus_pattern.search(line)
            if virus_match:
                virus_name = virus_match.group()

                if virus_name == 'A/Hong Kong/1550/02':
                    print("")

                line = line.replace(virus_name, "AAAAA")
                elem = line.split(" ")

                if virus_name not in col_names and len(elem) < 8:
                    continue

                elem = elem[-col_num:]
                data.append([virus_name] + elem)

        df = pd.DataFrame(data)
        df.columns = ['Test Virus'] + col_names

        long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

        print(long_df)
        virus_df = pd.concat([virus_df, long_df], ignore_index=True)


        page = pdf.pages[12]
        text = page.extract_text()

        text = re.sub(r'\n(\d+)\n', r' \1\n', text)
        text = text.replace("\n|", " *")
        text = re.sub(r'\n(?![AV])', ' ', text)
        text = text.replace("|", "*")
        text = text.replace("<", "<40")
        text = text.replace("Haemagglutination inhibition titre1 Post infection ferret sera ", "")
        text = text.replace(" 1. <40,<4040", "")
        lines = text.split('\n')

        virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')


        col_names = ['A/Panama/2007/99', 'A/New York/55/01', 'A/Egypt/130/02', 'A/Fujian/411/02', 'A/Finland/170/03', 'A/Wyoming/3/03', 'A/UK/1861/03']

        col_num = len(col_names)


        data = []
        for line in lines:
            virus_match = virus_pattern.search(line)
            if virus_match:
                virus_name = virus_match.group()

                if virus_name == 'A/Hong Kong/1550/02':
                    print("")

                line = line.replace(virus_name, "AAAAA")
                elem = line.split(" ")

                if virus_name not in col_names and len(elem) < 8:
                    continue

                elem = elem[-col_num:]
                data.append([virus_name] + elem)

        df = pd.DataFrame(data)
        df.columns = ['Test Virus'] + col_names

        long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

        print(long_df)
        virus_df = pd.concat([virus_df, long_df], ignore_index=True)

    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in  ['2005-feb.pdf', '2006-mar.pdf', '2006-sep.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            page_number = pdf.pages.index(page)
            if text and "Table" in text and "Antigenic analyses of influenza A H3N2 viruses" in text:
                text = text.replace("<", "<40")
                text = text.replace("WELL", "Well")
                text = text.replace("X-147", "A/Wy")
                text = text.replace("A/Bay/", "A/Bay")
                text = text.replace("A/Beryern/4/2006", "A/Bayern/4/2006")
                lines = text.split('\n')

                for line in lines:
                    if line.startswith("Viruses Isolation"):
                        header_line = line
                        break

                virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')
                full_virus_names = []

                for line in lines:
                    matches = virus_pattern.findall(line)
                    for m in matches:
                        if m not in full_virus_names:
                            full_virus_names.append(m)


                tokens = header_line.split()[2:]
                col_names = []

                for t in tokens:
                    matches = [full for full in full_virus_names if full.startswith(t)]

                    if matches:
                        col_names.append(matches[0])
                    else:
                        cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                        if cityname:
                            matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                            col_names.append(matches[0])

                col_num = len(col_names)

                if filename == '2005-feb.pdf' and page_number == 9:
                    col_names[9] = 'A/Victoria/513/2004'
                    col_names[10] = 'A/Victoria/110/2004'

                data = []
                for line in lines:
                    virus_match = virus_pattern.search(line)
                    if virus_match:
                        virus_name = virus_match.group()

                        elem = line.split(" ")

                        if virus_name not in col_names and len(elem) < 8:
                            continue

                        elem = elem[-col_num:]
                        data.append([virus_name] + elem)

                df = pd.DataFrame(data)
                df.columns = ['Test Virus'] + col_names

                long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in  ['2005-sep.pdf', '2007-mar.pdf', '2007-sep.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            page_number = pdf.pages.index(page)
            if text and "Table" in text and "Antigenic analyses of influenza A H3N2 viruses" in text:
                text = re.sub(r'\n(\d+)\n', r' \1\n', text)
                text = text.replace("\n|", " *")
                text = re.sub(r'\n(?![AVD])', ' ', text)
                text = text.replace("|", "*")
                text = text.replace("<", "<40")
                text = text.replace("Haemagglutination inhibition titre1 Post infection ferret sera 160", "160\n")
                text = text.replace(" Haemagglutination inhibition titre1 Post-infection ferret sera", "")
                text = text.replace(" Haemagglutination inhibition titre1 Post infection ferret sera", "")
                text = text.replace(" Haemaggutination inhibition titre1 Post infecton ferret sera", "")
                text = text.replace(" 1. <40,<4040", "")
                text = text.replace("A/Hong Kong/2657/2007 8.6.07 MDCK2\\1 <40 40 20 10 * 80 80 * <40 40 13", "A/Hong Kong/2657/2007 8.6.07 MDCK2\\1 <40 40 20 10 * 80 80 * <40 40")
                text = text.replace("A/Khabarovsk/528/07 9.4.07 C1\\1 40 40 80 80 40 20 20 <40 160 80 12", "A/Khabarovsk/528/07 9.4.07 C1\\1 40 40 80 80 40 20 20 <40 160 80")
                text = text.replace("A/Johannesburg/74/07 9.7.07 MDCKx\\1 80 160 320 80 80 160 <40 160 <40 160 14", "A/Johannesburg/74/07 9.7.07 MDCKx\\1 80 160 320 80 80 160 <40 160 <40 160")

                text = text.replace("A/Georgia/125/2007 26.03.2007 MDCK2\\1 160 160 80 160 80 160 40 160 160 160 15", "A/Georgia/125/2007 26.03.2007 MDCK2\\1 160 160 80 160 80 160 40 160 160 160")
                lines = text.split('\n')

                for line in lines:
                    if line.startswith("Viruses Isolation") or line.startswith("Viruses Collection"):
                        header_line = line
                        break

                virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')
                full_virus_names = []

                for line in lines:
                    matches = virus_pattern.findall(line)
                    for m in matches:
                        if m not in full_virus_names:
                            full_virus_names.append(m)


                tokens = header_line.split()[2:]
                col_names = []

                for t in tokens:
                    matches = [full for full in full_virus_names if full.startswith(t)]

                    if matches:
                        col_names.append(matches[0])
                    else:
                        cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                        if cityname:
                            matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                            col_names.append(matches[0])

                col_num = len(col_names)

                data = []
                for line in lines:
                    virus_match = virus_pattern.search(line)
                    if virus_match:
                        virus_name = virus_match.group()

                        elem = line.split(" ")

                        if virus_name not in col_names and len(elem) < 8:
                            continue

                        elem = elem[-col_num:]
                        data.append([virus_name] + elem)

                df = pd.DataFrame(data)
                df.columns = ['Test Virus'] + col_names

                long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in  ['2008-mar.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            page_number = pdf.pages.index(page)
            if text and "Table" in text and "Antigenic analyses of influenza A H3N2 viruses" in text:
                text = re.sub(r'\n(\d+)\n', r' \1\n', text)
                text = text.replace("\n|", " *")
                text = re.sub(r'\n(?![AVD])', ' ', text)
                text = text.replace("|", "*")
                text = text.replace("<", "<40")
                text = text.replace("Haemagglutination inhibition titre1 Post infection ferret sera 160", "160\n")
                text = text.replace(" Haemagglutination inhibition titre1 Post-infection ferret sera", "")
                text = text.replace(" Haemagglutination inhibition titre1 Post infection ferret sera", "")
                text = text.replace(" Haemaggutination inhibition titre1 Post infecton ferret sera", "")
                text = text.replace(" 1. <40,<4040", "")
                text = text.replace("A/Hong Kong/2657/2007 8.6.07 MDCK2\\1 <40 40 20 10 * 80 80 * <40 40 13", "A/Hong Kong/2657/2007 8.6.07 MDCK2\\1 <40 40 20 10 * 80 80 * <40 40")
                text = text.replace("A/Khabarovsk/528/07 9.4.07 C1\\1 40 40 80 80 40 20 20 <40 160 80 12", "A/Khabarovsk/528/07 9.4.07 C1\\1 40 40 80 80 40 20 20 <40 160 80")
                text = text.replace("A/Johannesburg/74/07 9.7.07 MDCKx\\1 80 160 320 80 80 160 <40 160 <40 160 14", "A/Johannesburg/74/07 9.7.07 MDCKx\\1 80 160 320 80 80 160 <40 160 <40 160")
                text = text.replace("A/Georgia/125/2007 26.03.2007 MDCK2\\1 160 160 80 160 80 160 40 160 160 160 15", "A/Georgia/125/2007 26.03.2007 MDCK2\\1 160 160 80 160 80 160 40 160 160 160")
                text = text.replace(" TEST VIRUSES", "")
                text = text.replace(" Passage History", "")
                text = text.replace(" Vaccine strain", "")
                
                lines = text.split('\n')

                for line in lines:
                    if line.startswith("Viruses Isolation") or line.startswith("Viruses Collection"):
                        header_line = line
                        break

                virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')
                full_virus_names = []

                for line in lines:
                    matches = virus_pattern.findall(line)
                    for m in matches:
                        if m not in full_virus_names:
                            full_virus_names.append(m)


                tokens = header_line.split()[2:]
                col_names = []

                for t in tokens:
                    matches = [full for full in full_virus_names if full.startswith(t)]

                    if matches:
                        col_names.append(matches[0])
                    else:
                        cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                        if cityname:
                            matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                            col_names.append(matches[0])

                if filename == '2008-mar.pdf':
                    col_names[6] = 'A/Trieste/25E/2007'
                    col_names[7] = 'A/Wisconsin/3/2007'
                    col_names[8] = 'A/Henan/Jinshui/147/2007'

                col_num = len(col_names)

                data = []
                for line in lines:
                    virus_match = virus_pattern.search(line)
                    if virus_match:
                        virus_name = virus_match.group()

                        elem = line.split(" ")

                        if virus_name not in col_names and len(elem) < 8:
                            continue

                        elem = elem[-col_num:]
                        data.append([virus_name] + elem)

                df = pd.DataFrame(data)
                df.columns = ['Test Virus'] + col_names

                long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in  ['2008-sep.pdf', '2009-feb.pdf', '2009-sep.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            page_number = pdf.pages.index(page)
            if text and "Table" in text and ("Antigenic analysis of influenza AH3N2 viruses" in text or "Antigenic analyses of influenza A H3N2 viruses" in text or "Antigenic analysis of influenza A H3N2 viruses" in text):
                text = text.replace("<", "<40")
                text = text.replace("ND", "*")
                text = text.replace("NT", "*")
                lines = text.split('\n')

                for line in lines:
                    if line.startswith("Viruses Collection Passage"):
                        tokens = line.split("Viruses Collection Passage ")[1].split(" ")
                        break
                    elif line.startswith("Collection Passage"):
                        tokens = line.split("Collection Passage ")[1].split(" ")
                        break

                virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')
                full_virus_names = []

                for line in lines:
                    matches = virus_pattern.findall(line)
                    for m in matches:
                        if m not in full_virus_names:
                            full_virus_names.append(m)


                
                col_names = []

                for t in tokens:
                    matches = [full for full in full_virus_names if full.startswith(t)]

                    if matches:
                        col_names.append(matches[0])
                    else:
                        cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                        if cityname:
                            matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                            col_names.append(matches[0])

                col_num = len(col_names)

                if filename == '2008-sep.pdf': 
                    if page_number == 11 or page_number == 12:
                        col_names[4] = 'A/Trieste/25E/2007'
                        col_names[5] = 'A/Wisconsin/3/2007'
                        col_names[6] = 'A/Henan/Jinshui/147/2007'
                    elif page_number == 13:
                        col_names[6] = 'A/Henan/Jinshui/147/2007'
                elif filename == '2009-sep.pdf':
                    if page_number == 17:
                        col_names[6] = 'A/Brisbane/24/2008'
                        col_names[8] = 'A/Hong Kong/1985/2009'
                        col_names[10] = 'A/Perth/16/2009'
                    elif page_number == 18:
                        col_names[6] = 'A/Hong Kong/1985/2009'
                        col_names[8] = 'A/Perth/16/2009'
                    elif page_number == 19:
                        col_names[7] = 'A/Hong Kong/1985/2009'
                        col_names[9] = 'A/Wisconsin/15/2009'
                elif filename == '2010-feb.pdf':
                    if page_number == 39:
                        col_names[6] = 'A/Hong Kong/1985/2009'
                        col_names[8] = 'A/Wisconsin/15/2009'

                

                data = []
                for line in lines:
                    virus_match = virus_pattern.search(line)
                    if virus_match:
                        virus_name = virus_match.group()

                        elem = line.split(" ")

                        if virus_name not in col_names and len(elem) < 8:
                            continue

                        elem = elem[-col_num:]
                        data.append([virus_name] + elem)

                df = pd.DataFrame(data)
                df.columns = ['Test Virus'] + col_names

                long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in  ['2010-feb.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            page_number = pdf.pages.index(page)
            if text and "Table" in text and ("Antigenic analysis of influenza AH3N2 viruses" in text or "Antigenic analyses of influenza A H3N2 viruses" in text or "Antigenic analysis of influenza A H3N2 viruses" in text):
                text = text.replace("<", "<40")
                text = text.replace("ND", "*")
                text = text.replace("NT", "*")
                lines = text.split('\n')

                for line in lines:
                    if line.startswith("Viruses Collection Passage"):
                        tokens = line.split("Viruses Collection Passage ")[1].split(" ")
                    elif line.startswith("Collection Passage"):
                        tokens = line.split("Collection Passage ")[1].split(" ")
                    elif line.startswith("Date History"):
                        datelist = line.split("Date History ")[1].split(" ")


                virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')
                full_virus_names = []

                for line in lines:
                    matches = virus_pattern.findall(line)
                    for m in matches:
                        if m not in full_virus_names:
                            full_virus_names.append(m)


                
                col_names = []

                for i, t in enumerate(tokens):
                    matches = [full for full in full_virus_names if full.startswith(t)]

                    if matches:
                        if len(matches) > 1:
                            tid = datelist[i].split("/")[0]
                            for vname in matches:
                                if tid in vname.split("/")[2]:
                                    col_names.append(vname)
                                    break
                        else:
                            col_names.append(matches[0])
                    else:
                        cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                        if cityname:
                            matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                            if len(matches) > 1:
                                tid = datelist[i].split("/")[0]
                                for vname in matches:
                                    if tid in vname.split("/")[2]:
                                        col_names.append(vname)
                                        break
                            else:
                                col_names.append(matches[0])

                col_num = len(col_names)

                

                data = []
                for line in lines:
                    virus_match = virus_pattern.search(line)
                    if virus_match:
                        virus_name = virus_match.group()

                        elem = line.split(" ")

                        if virus_name not in col_names and len(elem) < 8:
                            continue

                        elem = elem[-col_num:]
                        data.append([virus_name] + elem)

                df = pd.DataFrame(data)
                df.columns = ['Test Virus'] + col_names

                long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in  ['2010-sep.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            page_number = pdf.pages.index(page)
            if text and "Table" in text and ("Antigenic analysis of influenza AH3N2 viruses" in text or "Antigenic analysis of influenza A(H3N2) viruses" in text or  "Antigenic analyses of influenza A H3N2 viruses" in text or "Antigenic analysis of influenza A H3N2 viruses" in text):
                text = text.replace("15/0934430/09", "15/09 34430/09")
                text = text.replace(" Viruses Collection Passage", "\nViruses Collection Passage")
                text = text.replace("<", "<40")
                text = text.replace("ND", "*")
                text = text.replace("NT", "*")
                lines = text.split('\n')

                for line in lines:
                    if line.startswith("Viruses Collection Passage"):
                        tokens = line.split("Viruses Collection Passage ")[1].split(" ")
                    elif line.startswith("Collection Passage"):
                        tokens = line.split("Collection Passage ")[1].split(" ")
                    elif line.startswith("Date History"):
                        datelist = line.split("Date History ")[1].split(" ")


                virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')
                full_virus_names = []

                for line in lines:
                    matches = virus_pattern.findall(line)
                    for m in matches:
                        if m not in full_virus_names:
                            full_virus_names.append(m)


                
                col_names = []

                for i, t in enumerate(tokens):
                    matches = [full for full in full_virus_names if full.startswith(t)]

                    if matches:
                        if len(matches) > 1:
                            tid = datelist[i].split("/")[0]
                            for vname in matches:
                                if tid in vname.split("/")[2]:
                                    col_names.append(vname)
                                    break
                        else:
                            col_names.append(matches[0])
                    else:
                        if '/' in t:
                            cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                            if cityname:
                                matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                                if len(matches) > 1:
                                    tid = datelist[i].split("/")[0]
                                    for vname in matches:
                                        if tid in vname.split("/")[2]:
                                            col_names.append(vname)
                                            break
                                else:
                                    col_names.append(matches[0])




                data = []
                for line in lines:
                    virus_match = virus_pattern.search(line)
                    if virus_match:
                        virus_name = virus_match.group()
                        elem = line.split(" ")
                        elem = [s for s in elem if re.fullmatch(r'[0-9<\*]+', s)][:len(col_names)]
                        data.append([virus_name] + elem)

                df = pd.DataFrame(data)
                df.columns = ['Test Virus'] + col_names

                long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in  ['2011-feb.pdf', '2011-sep.pdf', '2012-feb.pdf', '2012-sep.pdf', '2013-feb.pdf', '2013-sep.pdf', '2014-feb.pdf', '2014-sep.pdf', '2015-feb.pdf', '2015-sep.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            page_number = pdf.pages.index(page)
            if text and "Table" in text and "Neutralisation titre" not in text and ("Antigenic analysis of influenza AH3N2 viruses" in text or "Antigenic analysis of influenza A(H3N2) viruses" in text or  "Antigenic analyses of influenza A H3N2 viruses" in text or "Antigenic analysis of influenza A H3N2 viruses" in text or "Antigenic analyses of influenza A(H3N2) viruses" in text):
                text = text.replace("15/0934430/09", "15/09 34430/09")
                text = text.replace(" Serbia ", " A/Serbia ")
                text = text.replace(" Viruses Collection Passage", "\nViruses Collection Passage")
                text = text.replace("<", "<40")
                text = text.replace(">", "")
                text = text.replace("X-199", "A/Rh")
                text = text.replace("NIB-85 (A/Almaty/2958/2013)", "A/Almaty/2958/2013")
                text = text.replace("NIB-85", "A/Alma")
                text = text.replace("A/PerthA/Alaska", "A/Perth A/Alaska")
                text = text.replace("A/Alabama5/2010", "A/Alabama/5/2010")
                text = text.replace("*A/Sth Afr", "A/Sou")
                text = text.replace("*A/Stock", "A/Stock")
                text = text.replace("A/Sth Afr", "A/Sou")
                text = text.replace("*A/Nor", "A/Nor")
                text = text.replace("*A/Switz", "A/Switz")
                text = text.replace("*", "")
                text = text.replace("ND", "*")
                text = text.replace("NT", "*")
                text = text.replace("2A/HK", "A/HK")
                text = text.replace("2A/Eng", "A/Eng")
                text = text.replace("A/Nth Carol", "A/North")
                text = text.replace(" cl123", "")
                text = text.replace(" cl121", "")
                text = text.replace("NYMC X-263B (A/HK/4801/2014)", "A/Hong Kong/4801/2014)")
                text = text.replace("NYMC X-261 (A/HK/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace("X-263B", "A/Hong")
                text = text.replace("X-261", "A/Hong")
                text = text.replace("HK/4801/14", "4801/14")
                text = text.replace("HK/7127/14", "7127/14")
                text = text.replace("NYMC X-263B (A/Hong Kong/4801/2014)", "A/Hong Kong/4801/2014")
                text = text.replace("NYMC X-261 (A/Hong Kong/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace("NIB-93 (A/Hong Kong/7127/2014)", "A/Hong Kong/7127/2014")

                lines = text.split('\n')

                for line in lines:
                    if line.startswith("Viruses Collection Passage plaques "):
                        tokens = line.split("Viruses Collection Passage plaques ")[1].split(" ")
                    elif line.startswith("Viruses Collection Passage"):
                        tokens = line.split("Viruses Collection Passage ")[1].split(" ")
                    elif line.startswith("Collection Passage"):
                        tokens = line.split("Collection Passage ")[1].split(" ")
                    elif line.startswith("Date History"):
                        datelist = line.split("Date History ")[1].split(" ")
                
                pattern = r'^A/.+/$'
                tokens = [s[:-1] if re.match(pattern, s) else s for s in tokens]

                if filename == '2014-sep.pdf':
                    if "A/Alma" in tokens:
                        datelist.insert(tokens.index("A/Alma"), "2958/13")

                if filename == '2015-sep.pdf':
                    if page_number == 58:
                        tokens[-2] = 'A/Hong'
                        tokens[-1] = 'A/Hong'
                        tokens[-5] = 'A/Hong'
                        datelist[-2] = '7127/14'
                        datelist[-1] = '7127/14'
                        datelist[-5] = '4801/14'




                assert len(tokens) == len(datelist)

                virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')
                full_virus_names = []

                for line in lines:
                    matches = virus_pattern.findall(line)
                    for m in matches:
                        if m not in full_virus_names:
                            full_virus_names.append(m)
                
                col_names = []

                for i, t in enumerate(tokens):
                    matches = [full for full in full_virus_names if full.startswith(t)]

                    if matches:
                        if len(matches) > 1:
                            tid = datelist[i].split("/")[0]
                            for vname in matches:
                                if tid in vname.split("/")[2]:
                                    col_names.append(vname)
                                    break
                        else:
                            col_names.append(matches[0])
                    else:
                        if '/' in t:
                            cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                            if cityname:
                                matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                                if len(matches) > 1:
                                    tid = datelist[i].split("/")[0]
                                    for vname in matches:
                                        if tid in vname.split("/")[2]:
                                            col_names.append(vname)
                                            break
                                else:
                                    col_names.append(matches[0])

                
                if filename == '2011-feb.pdf':
                    if page_number == 31:
                        col_names[2] = 'A/Hong Kong/34430/2009'
                if filename == '2011-sep.pdf':
                    if page_number == 41:
                        col_names.append("A/Wisconsin/15/2009")
                if filename == '2012-feb.pdf':
                    if page_number == 37:
                        col_names[2] = 'A/Perth/10/2009'
                if filename == '2015-feb.pdf':
                    if page_number == 53:
                        col_names.insert(0, "A/Hong Kong/5576/2014")
                        col_names.insert(0, "A/Hong Kong/7295/2014")
                if filename == '2015-sep.pdf':
                    if page_number == 55:
                        col_names.append("A/England/527/2014")
                        


                data = []
                for line in lines:
                    virus_match = virus_pattern.search(line)
                    if virus_match:
                        virus_name = virus_match.group()

                        def replacer(match):
                            a, b = int(match.group(1)), int(match.group(2))
                            if a % 10 == 0 and b % 10 == 0:
                                avg = (a + b) // 2
                                return str(avg)
                            else:
                                return match.group(0)

                        pattern = re.compile(r'(\d+)-(\d+)')
                        elem = pattern.sub(replacer, line).split(" ")

                        tmp = [s for s in elem if re.fullmatch(r'[0-9<\*]+', s)]
                        elem = []
                        for e in tmp:
                            if "<" in e or "*" in e:
                                elem.append(e)
                            elif int(e) % 10 == 0:
                                elem.append(e)

                        elem = elem[:len(col_names)]

                        if len(elem) == len(col_names):
                            data.append([virus_name] + elem)
                
                if len(data) > 0:
                    df = pd.DataFrame(data)
                    df.columns = ['Test Virus'] + col_names

                    long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                    virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in  ['2016-feb.pdf', '2016-sep.pdf', '2017-feb.pdf', '2017-sep.pdf', '2019-sep.pdf', '2020-feb.pdf', '2020-sep.pdf', '2022-feb.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            page_number = pdf.pages.index(page)
            if text and "Table" in text and "Neutralisation titre" not in text and "Neutralisation" not in text and ("Antigenic analysis of influenza AH3N2 viruses" in text or "Antigenic analysis of influenza A(H3N2) viruses" in text or  "Antigenic analyses of influenza A H3N2 viruses" in text or "Antigenic analysis of influenza A H3N2 viruses" in text or "Antigenic analyses of influenza A(H3N2) viruses" in text):
                text = text.replace("15/0934430/09", "15/09 34430/09")
                text = text.replace(" Serbia ", " A/Serbia ")
                text = text.replace(" Viruses Collection Passage", "\nViruses Collection Passage")
                text = text.replace("<", "<40")
                text = text.replace(">", "")
                text = text.replace("X-199", "A/Rh")
                text = text.replace("NIB-85 (A/Almaty/2958/2013)", "A/Almaty/2958/2013")
                text = text.replace("NIB-85", "A/Alma")
                text = text.replace("A/PerthA/Alaska", "A/Perth A/Alaska")
                text = text.replace("A/Alabama5/2010", "A/Alabama/5/2010")
                text = text.replace("*A/Sth Afr", "A/Sou")
                text = text.replace("A/Sth Africa", "A/Sou")
                text = text.replace("A/Sth Afr", "A/Sou")
                text = text.replace("A/S Africa", "A/Sou")
                text = text.replace("*A/Stock", "A/Stock")
                text = text.replace("*A/Nor", "A/Nor")
                text = text.replace("*A/Switz", "A/Switz")
                text = text.replace("*", "")
                text = text.replace("ND", "*")
                text = text.replace("NT", "*")
                text = text.replace("2A/HK", "A/HK")
                text = text.replace("2A/Eng", "A/Eng")
                text = text.replace("A/Nth Carol", "A/North")
                text = text.replace(" cl123", "")
                text = text.replace(" cl121", "")
                text = text.replace("NYMC X-263B (A/HK/4801/2014)", "A/Hong Kong/4801/2014)")
                text = text.replace("NYMC X-261 (A/HK/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace("X-263B", "A/Hong")
                text = text.replace("X-261", "A/Hong")
                text = text.replace("HK/4801/14", "4801/14")
                text = text.replace("HK/7127/14", "7127/14")
                text = text.replace("NYMC X-263B (A/Hong Kong/4801/2014)", "A/Hong Kong/4801/2014")
                text = text.replace("NYMC X-261 (A/Hong Kong/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace("NIB-93 (A/Hong Kong/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace(" Georgia", " A/Georgia")
                text = text.replace("plaq 20", "plaq")
                text = text.replace("A/Georgia HA1 substitutions", "A/Georgia\n HA1 substitutions")
                text = text.replace("532/15 Egg adaptation", "532/15\n Egg adaptation")
                text = text.replace("532/15 Kong/4801/2014", "532/15\n Kong/4801/2014")
                text = text.replace("532/15 A/Hong Kong/4801/2014", "532/15\n A/Hong Kong/4801/2014")
                text = text.replace("CBER08", "A/Nor")
                text = text.replace("C1.4", "3806/16")
                text = text.replace("NIB-103 (A/Norway/3806/2016)", "A/Norway/3806/2016")
                text = text.replace("NIB-103", "A/Norway/3806/2016")
                text = text.replace("A/La Rioja", "A/La")
                text = text.replace("NYMC X-327 (A/Kansas/14/17)", "A/Kansas/14/2017")
                text = text.replace("NYMC X-327", "A/Kans")
                text = text.replace("A/Kans/14/17", "14/17")
                text = text.replace("A/C'church", "A/Christ")
                text = text.replace("IVR-197  (A/South Australia/34/2019)", "A/South Australia/34/2019")
                text = text.replace("IVR-197", "A/South")
                text = text.replace("A/Sth Aus/34/19", "34/19")
                text = text.replace("A/Sth Aus", "A/South")

                if filename == '2020-feb.pdf' and page_number == 67:
                    continue
                if filename == '2020-sep.pdf' and page_number == 67:
                    continue



                lines = text.split('\n')

                for line in lines:
                    if line.startswith("Viruses Collection Passage plaques "):
                        tokens = line.split("Viruses Collection Passage plaques ")[1].split(" ")
                    elif line.startswith("Viruses Collection Passage"):
                        tokens = line.split("Viruses Collection Passage ")[1].split(" ")
                    elif line.startswith("Viruses Other Collection Passage"):
                        tokens = line.split("Viruses Other Collection Passage ")[1].split(" ")
                    elif line.startswith("Collection Passage"):
                        tokens = line.split("Collection Passage ")[1].split(" ")
                    elif line.startswith("Date History"):
                        datelist = line.split("Date History ")[1].split(" ")
                    elif line.startswith("information date history"):
                        datelist = line.split("information date history ")[1].split(" ")
                
                pattern = r'^A/.+/$'
                tokens = [s[:-1] if re.match(pattern, s) else s for s in tokens]

                if filename == '2014-sep.pdf':
                    if "A/Alma" in tokens:
                        datelist.insert(tokens.index("A/Alma"), "2958/13")

                if filename == '2015-sep.pdf':
                    if page_number == 58:
                        tokens[-2] = 'A/Hong'
                        tokens[-1] = 'A/Hong'
                        tokens[-5] = 'A/Hong'
                        datelist[-2] = '7127/14'
                        datelist[-1] = '7127/14'
                        datelist[-5] = '4801/14'

                if filename == '2017-sep.pdf':
                    if page_number == 68:
                        datelist.append("3806/16")
                    elif page_number == 69:
                        datelist.insert(-2, "3806/16")

                assert len(tokens) == len(datelist)

                virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')
                full_virus_names = []

                for line in lines:
                    matches = virus_pattern.findall(line)
                    for m in matches:
                        if m not in full_virus_names:
                            full_virus_names.append(m)
                

                if filename == '2016-sep.pdf':
                    if page_number == 63:
                        full_virus_names.append("A/Slovenia/3188/2015")
                if filename == '2017-feb.pdf':
                    if page_number == 48 or page_number == 49:
                        full_virus_names.append("A/Oman/2585/2016")
                        full_virus_names.append("A/Norway/4436/2016")

                if filename == '2017-sep.pdf' or filename == '2018-feb.pdf':
                    full_virus_names.append("A/Oman/2585/2016")
                    full_virus_names.append("A/Norway/4436/2016")
                    full_virus_names.append("A/Norway/4465/2016")
                    full_virus_names.append("A/Greece/4/2017")

                full_virus_names.append("A/La Rioja/2202/2018")
                full_virus_names.append("A/Norway/3275/2018")
                full_virus_names.append("A/Hong Kong/2671/2019")


                col_names = []

                for i, t in enumerate(tokens):
                    matches = [full for full in full_virus_names if full.startswith(t)]

                    if matches:
                        if len(matches) > 1:
                            tid = datelist[i].split("/")[0]
                            for vname in matches:
                                if tid in vname.split("/")[2]:
                                    col_names.append(vname)
                                    break
                        else:
                            col_names.append(matches[0])
                    else:
                        if '/' in t:
                            cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                            if cityname:
                                matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                                if len(matches) > 1:
                                    tid = datelist[i].split("/")[0]
                                    for vname in matches:
                                        if tid in vname.split("/")[2]:
                                            col_names.append(vname)
                                            break
                                else:
                                    col_names.append(matches[0])

                
                if filename == '2011-feb.pdf':
                    if page_number == 31:
                        col_names[2] = 'A/Hong Kong/34430/2009'
                if filename == '2011-sep.pdf':
                    if page_number == 41:
                        col_names.append("A/Wisconsin/15/2009")
                if filename == '2012-feb.pdf':
                    if page_number == 37:
                        col_names[2] = 'A/Perth/10/2009'
                if filename == '2015-feb.pdf':
                    if page_number == 53:
                        col_names.insert(0, "A/Hong Kong/5576/2014")
                        col_names.insert(0, "A/Hong Kong/7295/2014")
                if filename == '2015-sep.pdf':
                    if page_number == 55:
                        col_names.append("A/England/527/2014")

                        


                data = []
                for line in lines:
                    virus_match = virus_pattern.search(line)
                    if virus_match:
                        virus_name = virus_match.group()

                        def replacer(match):
                            a, b = int(match.group(1)), int(match.group(2))
                            if a % 10 == 0 and b % 10 == 0:
                                avg = (a + b) // 2
                                return str(avg)
                            else:
                                return match.group(0)

                        pattern = re.compile(r'(\d+)-(\d+)')
                        elem = pattern.sub(replacer, line).split(" ")

                        tmp = [s for s in elem if re.fullmatch(r'[0-9<\*]+', s)]
                        elem = []
                        for e in tmp:
                            if "<" in e or "*" in e:
                                elem.append(e)
                            elif int(e) % 10 == 0:
                                elem.append(e)

                        elem = elem[:len(col_names)]

                        if len(elem) == len(col_names):
                            data.append([virus_name] + elem)
                
                if len(data) > 0:
                    df = pd.DataFrame(data)
                    df.columns = ['Test Virus'] + col_names


                    long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                    virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in  ['2022-sep.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            page_number = pdf.pages.index(page)
            if text and "Table" in text and "Neutralisation titre" not in text and "Neutralisation" not in text and ("Antigenic analysis of influenza AH3N2 viruses" in text or "Antigenic analysis of influenza A(H3N2) viruses" in text or  "Antigenic analyses of influenza A H3N2 viruses" in text or "Antigenic analysis of influenza A H3N2 viruses" in text or "Antigenic analyses of influenza A(H3N2) viruses" in text):
                text = text.replace("15/0934430/09", "15/09 34430/09")
                text = text.replace(" Serbia ", " A/Serbia ")
                text = text.replace(" Viruses Collection Passage", "\nViruses Collection Passage")
                text = text.replace("<", "<40")
                text = text.replace(">", "")
                text = text.replace("X-199", "A/Rh")
                text = text.replace("NIB-85 (A/Almaty/2958/2013)", "A/Almaty/2958/2013")
                text = text.replace("NIB-85", "A/Alma")
                text = text.replace("A/PerthA/Alaska", "A/Perth A/Alaska")
                text = text.replace("A/Alabama5/2010", "A/Alabama/5/2010")
                text = text.replace("*A/Sth Afr", "A/Sou")
                text = text.replace("A/Sth Africa", "A/Sou")
                text = text.replace("A/Sth Afr", "A/Sou")
                text = text.replace("A/S Africa", "A/Sou")
                text = text.replace("*A/Stock", "A/Stock")
                text = text.replace("*A/Nor", "A/Nor")
                text = text.replace("*A/Switz", "A/Switz")
                text = text.replace("*", "")
                text = text.replace("ND", "*")
                text = text.replace("NT", "*")
                text = text.replace("2A/HK", "A/HK")
                text = text.replace("2A/Eng", "A/Eng")
                text = text.replace("A/Nth Carol", "A/North")
                text = text.replace(" cl123", "")
                text = text.replace(" cl121", "")
                text = text.replace("NYMC X-263B (A/HK/4801/2014)", "A/Hong Kong/4801/2014)")
                text = text.replace("NYMC X-261 (A/HK/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace("X-263B", "A/Hong")
                text = text.replace("X-261", "A/Hong")
                text = text.replace("HK/4801/14", "4801/14")
                text = text.replace("HK/7127/14", "7127/14")
                text = text.replace("NYMC X-263B (A/Hong Kong/4801/2014)", "A/Hong Kong/4801/2014")
                text = text.replace("NYMC X-261 (A/Hong Kong/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace("NIB-93 (A/Hong Kong/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace(" Georgia", " A/Georgia")
                text = text.replace("plaq 20", "plaq")
                text = text.replace("A/Georgia HA1 substitutions", "A/Georgia\n HA1 substitutions")
                text = text.replace("532/15 Egg adaptation", "532/15\n Egg adaptation")
                text = text.replace("532/15 Kong/4801/2014", "532/15\n Kong/4801/2014")
                text = text.replace("532/15 A/Hong Kong/4801/2014", "532/15\n A/Hong Kong/4801/2014")
                text = text.replace("CBER08", "A/Nor")
                text = text.replace("C1.4", "3806/16")
                text = text.replace("NIB-103 (A/Norway/3806/2016)", "A/Norway/3806/2016")
                text = text.replace("NIB-103", "A/Norway/3806/2016")
                text = text.replace("NYMC X-327 (A/Kansas/14/17)", "A/Kansas/14/2017")
                text = text.replace("NYMC X-327", "A/Kans")
                text = text.replace("A/Kans/14/17", "14/17")
                text = text.replace("A/C'church", "A/Christ")
                text = text.replace("IVR-197  (A/South Australia/34/2019)", "A/South Australia/34/2019")
                text = text.replace("IVR-197", "A/South")
                text = text.replace("A/Sth Aus/34/19", "34/19")
                text = text.replace("A/Sth Aus", "A/South")

                if filename == '2020-feb.pdf' and page_number == 67:
                    continue
                if filename == '2020-sep.pdf' and page_number == 67:
                    continue



                lines = text.split('\n')

                for line in lines:
                    if line.startswith("Viruses Collection Passage plaques "):
                        tokens = line.split("Viruses Collection Passage plaques ")[1].split(" ")
                    elif line.startswith("Viruses Collection Passage"):
                        tokens = line.split("Viruses Collection Passage ")[1].split(" ")
                    elif line.startswith("Viruses Other Collection Passage"):
                        tokens = line.split("Viruses Other Collection Passage ")[1].split(" ")
                    elif line.startswith("Collection Passage"):
                        tokens = line.split("Collection Passage ")[1].split(" ")
                    elif line.startswith("Date History"):
                        datelist = line.split("Date History ")[1].split(" ")
                    elif line.startswith("information date history"):
                        datelist = line.split("information date history ")[1].split(" ")
                    elif line.startswith("date history"):
                        datelist = line.split("date history ")[1].split(" ")
                
                pattern = r'^A/.+/$'
                tokens = [s[:-1] if re.match(pattern, s) else s for s in tokens]

                if filename == '2014-sep.pdf':
                    if "A/Alma" in tokens:
                        datelist.insert(tokens.index("A/Alma"), "2958/13")

                if filename == '2015-sep.pdf':
                    if page_number == 58:
                        tokens[-2] = 'A/Hong'
                        tokens[-1] = 'A/Hong'
                        tokens[-5] = 'A/Hong'
                        datelist[-2] = '7127/14'
                        datelist[-1] = '7127/14'
                        datelist[-5] = '4801/14'

                if filename == '2017-sep.pdf':
                    if page_number == 68:
                        datelist.append("3806/16")
                    elif page_number == 69:
                        datelist.insert(-2, "3806/16")

                assert len(tokens) == len(datelist)

                virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')
                full_virus_names = []

                for line in lines:
                    matches = virus_pattern.findall(line)
                    for m in matches:
                        if m not in full_virus_names:
                            full_virus_names.append(m)
                

                if filename == '2016-sep.pdf':
                    if page_number == 63:
                        full_virus_names.append("A/Slovenia/3188/2015")
                if filename == '2017-feb.pdf':
                    if page_number == 48 or page_number == 49:
                        full_virus_names.append("A/Oman/2585/2016")
                        full_virus_names.append("A/Norway/4436/2016")

                if filename == '2017-sep.pdf' or filename == '2018-feb.pdf':
                    full_virus_names.append("A/Oman/2585/2016")
                    full_virus_names.append("A/Norway/4436/2016")
                    full_virus_names.append("A/Norway/4465/2016")
                    full_virus_names.append("A/Greece/4/2017")

                full_virus_names.append("A/La Rioja/2202/2018")
                full_virus_names.append("A/Norway/3275/2018")
                full_virus_names.append("A/Hong Kong/2671/2019")


                col_names = []

                for i, t in enumerate(tokens):
                    matches = [full for full in full_virus_names if full.startswith(t)]

                    if matches:
                        if len(matches) > 1:
                            tid = datelist[i].split("/")[0]
                            for vname in matches:
                                if tid in vname.split("/")[2]:
                                    col_names.append(vname)
                                    break
                        else:
                            col_names.append(matches[0])
                    else:
                        if '/' in t:
                            cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                            if cityname:
                                matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                                if len(matches) > 1:
                                    tid = datelist[i].split("/")[0]
                                    for vname in matches:
                                        if tid in vname.split("/")[2]:
                                            col_names.append(vname)
                                            break
                                else:
                                    col_names.append(matches[0])

                
                if filename == '2011-feb.pdf':
                    if page_number == 31:
                        col_names[2] = 'A/Hong Kong/34430/2009'
                if filename == '2011-sep.pdf':
                    if page_number == 41:
                        col_names.append("A/Wisconsin/15/2009")
                if filename == '2012-feb.pdf':
                    if page_number == 37:
                        col_names[2] = 'A/Perth/10/2009'
                if filename == '2015-feb.pdf':
                    if page_number == 53:
                        col_names.insert(0, "A/Hong Kong/5576/2014")
                        col_names.insert(0, "A/Hong Kong/7295/2014")
                if filename == '2015-sep.pdf':
                    if page_number == 55:
                        col_names.append("A/England/527/2014")

                        


                data = []
                for line in lines:
                    virus_match = virus_pattern.search(line)
                    if virus_match:
                        virus_name = virus_match.group()

                        def replacer(match):
                            a, b = int(match.group(1)), int(match.group(2))
                            if a % 10 == 0 and b % 10 == 0:
                                avg = (a + b) // 2
                                return str(avg)
                            else:
                                return match.group(0)

                        pattern = re.compile(r'(\d+)-(\d+)')
                        elem = pattern.sub(replacer, line).split(" ")

                        tmp = [s for s in elem if re.fullmatch(r'[0-9<\*]+', s)]
                        elem = []
                        for e in tmp:
                            if "<" in e or "*" in e:
                                elem.append(e)
                            elif int(e) % 10 == 0:
                                elem.append(e)

                        elem = elem[:len(col_names)]

                        if len(elem) == len(col_names):
                            data.append([virus_name] + elem)
                
                if len(data) > 0:
                    df = pd.DataFrame(data)
                    df.columns = ['Test Virus'] + col_names


                    long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                    virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in  ['2023-feb.pdf', '2023-sep.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            page_number = pdf.pages.index(page)
            if text and "Table" in text and "Neutralisation titre" not in text and "Neutralisation" not in text and ("Antigenic analysis of influenza AH3N2 viruses" in text or "Antigenic analysis of influenza A(H3N2) viruses" in text or  "Antigenic analyses of influenza A H3N2 viruses" in text or "Antigenic analysis of influenza A H3N2 viruses" in text or "Antigenic analyses of influenza A(H3N2) viruses" in text):
                text = text.replace("15/0934430/09", "15/09 34430/09")
                text = text.replace(" Serbia ", " A/Serbia ")
                text = text.replace(" Viruses Collection Passage", "\nViruses Collection Passage")
                text = text.replace(">", "")
                text = text.replace("X-199", "A/Rh")
                text = text.replace("NIB-85 (A/Almaty/2958/2013)", "A/Almaty/2958/2013")
                text = text.replace("NIB-85", "A/Alma")
                text = text.replace("A/PerthA/Alaska", "A/Perth A/Alaska")
                text = text.replace("A/Alabama5/2010", "A/Alabama/5/2010")
                text = text.replace("*A/Sth Afr", "A/Sou")
                text = text.replace("A/Sth Africa", "A/Sou")
                text = text.replace("A/Sth Afr", "A/Sou")
                text = text.replace("A/S Africa", "A/Sou")
                text = text.replace("*A/Stock", "A/Stock")
                text = text.replace("*A/Nor", "A/Nor")
                text = text.replace("*A/Switz", "A/Switz")
                text = text.replace("*", "")
                text = text.replace("ND", "*")
                text = text.replace("NT", "*")
                text = text.replace("2A/HK", "A/HK")
                text = text.replace("2A/Eng", "A/Eng")
                text = text.replace("A/Nth Carol", "A/North")
                text = text.replace(" cl123", "")
                text = text.replace(" cl121", "")
                text = text.replace("NYMC X-263B (A/HK/4801/2014)", "A/Hong Kong/4801/2014)")
                text = text.replace("NYMC X-261 (A/HK/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace("X-263B", "A/Hong")
                text = text.replace("X-261", "A/Hong")
                text = text.replace("HK/4801/14", "4801/14")
                text = text.replace("HK/7127/14", "7127/14")
                text = text.replace("NYMC X-263B (A/Hong Kong/4801/2014)", "A/Hong Kong/4801/2014")
                text = text.replace("NYMC X-261 (A/Hong Kong/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace("NIB-93 (A/Hong Kong/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace(" Georgia", " A/Georgia")
                text = text.replace("plaq 20", "plaq")
                text = text.replace("A/Georgia HA1 substitutions", "A/Georgia\n HA1 substitutions")
                text = text.replace("532/15 Egg adaptation", "532/15\n Egg adaptation")
                text = text.replace("532/15 Kong/4801/2014", "532/15\n Kong/4801/2014")
                text = text.replace("532/15 A/Hong Kong/4801/2014", "532/15\n A/Hong Kong/4801/2014")
                text = text.replace("CBER08", "A/Nor")
                text = text.replace("C1.4", "3806/16")
                text = text.replace("NIB-103 (A/Norway/3806/2016)", "A/Norway/3806/2016")
                text = text.replace("NIB-103", "A/Norway/3806/2016")
                text = text.replace("NYMC X-327 (A/Kansas/14/17)", "A/Kansas/14/2017")
                text = text.replace("NYMC X-327", "A/Kans")
                text = text.replace("A/Kans/14/17", "14/17")
                text = text.replace("A/C'church", "A/Christ")
                text = text.replace("IVR-197  (A/South Australia/34/2019)", "A/South Australia/34/2019")
                text = text.replace("IVR-197", "A/South")
                text = text.replace("A/Sth Aus/34/19", "34/19")
                text = text.replace("A/Sth Aus", "A/South")
                text = text.replace("A/DarwinA/NorwayA/NorwayA/Poland","A/Darwin A/Norway A/Norway A/Poland")
                text = text.replace("A/DarwinA/NorwayA/Norway", "A/Darwin A/Norway A/Norway")
                text = text.replace("A/NorwayA/NorwayA/Poland", "A/Norway A/Norway A/Poland")
                text = text.replace("A/CatalA/Brandenburg", "A/Catal A/Brandenburg")

                if filename == '2020-feb.pdf' and page_number == 67:
                    continue
                if filename == '2020-sep.pdf' and page_number == 67:
                    continue

                lines = text.split('\n')

                tokens = []
                datelist = []

                for line in lines:
                    if line.startswith("Viruses Collection Passage plaques "):
                        tokens = line.split("Viruses Collection Passage plaques ")[1].split(" ")
                    elif line.startswith("Viruses Collection Passage"):
                        tokens = line.split("Viruses Collection Passage ")[1].split(" ")
                    elif line.startswith("Viruses Other Collection Passage"):
                        tokens = line.split("Viruses Other Collection Passage ")[1].split(" ")
                    elif line.startswith("Collection Passage"):
                        tokens = line.split("Collection Passage ")[1].split(" ")
                    elif line.startswith("Date History"):
                        datelist = line.split("Date History ")[1].split(" ")
                    elif line.startswith("information date history"):
                        datelist = line.split("information date history ")[1].split(" ")
                    elif line.startswith("date history"):
                        datelist = line.split("date history ")[1].split(" ")
                
                pattern = r'^A/.+/$'
                tokens = [s[:-1] if re.match(pattern, s) else s for s in tokens]

                if filename == '2014-sep.pdf':
                    if "A/Alma" in tokens:
                        datelist.insert(tokens.index("A/Alma"), "2958/13")

                if filename == '2015-sep.pdf':
                    if page_number == 58:
                        tokens[-2] = 'A/Hong'
                        tokens[-1] = 'A/Hong'
                        tokens[-5] = 'A/Hong'
                        datelist[-2] = '7127/14'
                        datelist[-1] = '7127/14'
                        datelist[-5] = '4801/14'

                if filename == '2017-sep.pdf':
                    if page_number == 68:
                        datelist.append("3806/16")
                    elif page_number == 69:
                        datelist.insert(-2, "3806/16")


                if filename == '2023-feb.pdf':
                    if page_number == 67:
                        datelist = ['925256/20',
                                    'e0826360/20',
                                    '10/22',
                                    '5/21',
                                    '9/21',
                                    '24873/21',
                                    '24873/21',
                                    '1724/22',
                                    '97/22',
                                    '8720/2022',
                                    '99/22',
                                    '28542/22',
                                    'NSVH-2067/22']
                    elif page_number == 68 or page_number == 70 or page_number == 71 or page_number == 75 or page_number == 76:
                        datelist  = ['925256/20',
                                    'e0826360/20',
                                    '10/22',
                                    '5/21',
                                    '9/21',
                                    '24873/21',
                                    '24873/21',
                                    '97/22',
                                    '8720/2022',
                                    'NSVH-2067/22']
                    elif page_number == 77:
                        datelist = ['925256/20',
                                    'e0826360/20',
                                    '10/22',
                                    '5/21',
                                    '5/21',
                                    '9/21',
                                    '24873/21',
                                    '24873/21',
                                    '97/22',
                                    '8720/2022',
                                    '50053/2022',
                                    'NSVH-2067/22']
                    elif page_number == 78:
                        continue

                if filename == '2023-sep.pdf':
                    if page_number == 68:
                        tokens = ['A/Thuringen', 'A/Stockholm', 'A/Darwin', 'A/Norway', 'A/Norway', 'A/Slovenia', 'A/Lille', 'A/Catal', 'A/Albania']
                        datelist = ['10/22', '5/21', '9/21', '24873/21', '24873/21', '8720/22', '50053/22', 'NSVH-2067/22', '289813/2022']
                    elif page_number == 69:
                        tokens = ['A/Thuringen', 'A/Stockholm', 'A/Darwin', 'A/Norway', 'A/Norway', 'A/Slovenia', 'A/Lille', 'A/Catal', 'A/Albania', 'A/Albania', 'A/Brandenburg']
                        datelist = ['10/22', '5/21', '9/21', '24873/21', '24873/21', '8720/22', '50053/22', 'NSVH-2067/22', '289813/2022', '289813/2022', '15/2022']
                    elif page_number == 70:
                        tokens = ['A/Thuringen', 'A/Stockholm', 'A/Darwin', 'A/Norway', 'A/Norway', 'A/Slovenia', 'A/Lille', 'A/Catal', 'A/Albania', 'A/Albania', 'A/Albania', 'A/Albania']
                        datelist = ['10/22', '5/21', '9/21', '24873/21', '24873/21', '8720/22', '50053/22', 'NSVH-2067/22', '289813/2022', '289813/2022', '290270/2022', '290243/2022']
                    elif page_number == 71:
                        tokens = ['A/Thuringen', 'A/Stockholm', 'A/Darwin', 'A/Norway', 'A/Norway', 'A/Slovenia', 'A/Lille', 'A/Catal', 'A/Albania', 'A/Albania', 'A/Brandenburg']
                        datelist = ['10/22', '5/21', '9/21', '24873/21', '24873/21', '8720/22', '50053/22', 'NSVH-2067/22', '289813/2022', '289813/2022', '15/2022']


                assert len(tokens) == len(datelist)

                virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')
                full_virus_names = []

                for line in lines:
                    matches = virus_pattern.findall(line)
                    for m in matches:
                        if m not in full_virus_names:
                            full_virus_names.append(m)
                

                if filename == '2016-sep.pdf':
                    if page_number == 63:
                        full_virus_names.append("A/Slovenia/3188/2015")
                if filename == '2017-feb.pdf':
                    if page_number == 48 or page_number == 49:
                        full_virus_names.append("A/Oman/2585/2016")
                        full_virus_names.append("A/Norway/4436/2016")

                if filename == '2017-sep.pdf' or filename == '2018-feb.pdf':
                    full_virus_names.append("A/Oman/2585/2016")
                    full_virus_names.append("A/Norway/4436/2016")
                    full_virus_names.append("A/Norway/4465/2016")
                    full_virus_names.append("A/Greece/4/2017")

                full_virus_names.append("A/La Rioja/2202/2018")
                full_virus_names.append("A/Norway/3275/2018")
                full_virus_names.append("A/Hong Kong/2671/2019")
                full_virus_names.append("A/Catalonia/NSVH-2067/22")


                col_names = []

                for i, t in enumerate(tokens):
                    matches = [full for full in full_virus_names if full.startswith(t)]

                    if matches:
                        if len(matches) > 1:
                            tid = datelist[i].split("/")[0]
                            for vname in matches:
                                if tid in vname.split("/")[2]:
                                    col_names.append(vname)
                                    break
                        else:
                            col_names.append(matches[0])
                    else:
                        if '/' in t:
                            cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                            if cityname:
                                matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                                if len(matches) > 1:
                                    tid = datelist[i].split("/")[0]
                                    for vname in matches:
                                        if tid in vname.split("/")[2]:
                                            col_names.append(vname)
                                            break
                                else:
                                    col_names.append(matches[0])

                
                if filename == '2011-feb.pdf':
                    if page_number == 31:
                        col_names[2] = 'A/Hong Kong/34430/2009'
                if filename == '2011-sep.pdf':
                    if page_number == 41:
                        col_names.append("A/Wisconsin/15/2009")
                if filename == '2012-feb.pdf':
                    if page_number == 37:
                        col_names[2] = 'A/Perth/10/2009'
                if filename == '2015-feb.pdf':
                    if page_number == 53:
                        col_names.insert(0, "A/Hong Kong/5576/2014")
                        col_names.insert(0, "A/Hong Kong/7295/2014")
                if filename == '2015-sep.pdf':
                    if page_number == 55:
                        col_names.append("A/England/527/2014")



                data = []
                for line in lines:
                    virus_match = virus_pattern.search(line)
                    if virus_match:
                        virus_name = virus_match.group()

                        def replacer(match):
                            a, b = int(match.group(1)), int(match.group(2))
                            if a % 10 == 0 and b % 10 == 0:
                                avg = (a + b) // 2
                                return str(avg)
                            else:
                                return match.group(0)

                        pattern = re.compile(r'(\d+)-(\d+)')
                        elem = pattern.sub(replacer, line).split(" ")

                        tmp = [s for s in elem if re.fullmatch(r'[0-9<\*]+', s)]
                        elem = []
                        for e in tmp:
                            if "<" in e or "*" in e:
                                elem.append(e)
                            elif int(e) % 10 == 0:
                                elem.append(e)

                        elem = elem[:len(col_names)]

                        if len(elem) == len(col_names):
                            data.append([virus_name] + elem)
                
                if len(data) > 0:
                    df = pd.DataFrame(data)
                    df.columns = ['Test Virus'] + col_names


                    long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                    virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in  ['2023-feb.pdf', '2023-sep.pdf', '2024-feb.pdf', '2024-sep.pdf', '2025-feb.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            page_number = pdf.pages.index(page)
            if text and "Table" in text and "Neutralisation titre" not in text and "Neutralisation" not in text and ("Antigenic analysis of influenza AH3N2 viruses" in text or "Antigenic analysis of influenza A(H3N2) viruses" in text or  "Antigenic analyses of influenza A H3N2 viruses" in text or "Antigenic analysis of influenza A H3N2 viruses" in text or "Antigenic analyses of influenza A(H3N2) viruses" in text):
                text = text.replace("15/0934430/09", "15/09 34430/09")
                text = text.replace(" Serbia ", " A/Serbia ")
                text = text.replace(" Viruses Collection Passage", "\nViruses Collection Passage")
                text = text.replace(">", "")
                text = text.replace("X-199", "A/Rh")
                text = text.replace("NIB-85 (A/Almaty/2958/2013)", "A/Almaty/2958/2013")
                text = text.replace("NIB-85", "A/Alma")
                text = text.replace("A/PerthA/Alaska", "A/Perth A/Alaska")
                text = text.replace("A/Alabama5/2010", "A/Alabama/5/2010")
                text = text.replace("*A/Sth Afr", "A/Sou")
                text = text.replace("A/Sth Africa", "A/Sou")
                text = text.replace("A/Sth Afr", "A/Sou")
                text = text.replace("A/S Africa", "A/Sou")
                text = text.replace("*A/Stock", "A/Stock")
                text = text.replace("*A/Nor", "A/Nor")
                text = text.replace("*A/Switz", "A/Switz")
                text = text.replace("*", "")
                text = text.replace("ND", "*")
                text = text.replace("NT", "*")
                text = text.replace("2A/HK", "A/HK")
                text = text.replace("2A/Eng", "A/Eng")
                text = text.replace("A/Nth Carol", "A/North")
                text = text.replace(" cl123", "")
                text = text.replace(" cl121", "")
                text = text.replace("NYMC X-263B (A/HK/4801/2014)", "A/Hong Kong/4801/2014)")
                text = text.replace("NYMC X-261 (A/HK/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace("X-263B", "A/Hong")
                text = text.replace("X-261", "A/Hong")
                text = text.replace("HK/4801/14", "4801/14")
                text = text.replace("HK/7127/14", "7127/14")
                text = text.replace("NYMC X-263B (A/Hong Kong/4801/2014)", "A/Hong Kong/4801/2014")
                text = text.replace("NYMC X-261 (A/Hong Kong/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace("NIB-93 (A/Hong Kong/7127/2014)", "A/Hong Kong/7127/2014")
                text = text.replace(" Georgia", " A/Georgia")
                text = text.replace("plaq 20", "plaq")
                text = text.replace("A/Georgia HA1 substitutions", "A/Georgia\n HA1 substitutions")
                text = text.replace("532/15 Egg adaptation", "532/15\n Egg adaptation")
                text = text.replace("532/15 Kong/4801/2014", "532/15\n Kong/4801/2014")
                text = text.replace("532/15 A/Hong Kong/4801/2014", "532/15\n A/Hong Kong/4801/2014")
                text = text.replace("CBER08", "A/Nor")
                text = text.replace("C1.4", "3806/16")
                text = text.replace("NIB-103 (A/Norway/3806/2016)", "A/Norway/3806/2016")
                text = text.replace("NIB-103", "A/Norway/3806/2016")
                text = text.replace("NYMC X-327 (A/Kansas/14/17)", "A/Kansas/14/2017")
                text = text.replace("NYMC X-327", "A/Kans")
                text = text.replace("A/Kans/14/17", "14/17")
                text = text.replace("A/C'church", "A/Christ")
                text = text.replace("IVR-197  (A/South Australia/34/2019)", "A/South Australia/34/2019")
                text = text.replace("IVR-197", "A/South")
                text = text.replace("A/Sth Aus/34/19", "34/19")
                text = text.replace("A/Sth Aus", "A/South")
                text = text.replace("A/DarwinA/NorwayA/NorwayA/Poland","A/Darwin A/Norway A/Norway A/Poland")
                text = text.replace("A/DarwinA/NorwayA/Norway", "A/Darwin A/Norway A/Norway")
                text = text.replace("A/NorwayA/NorwayA/Poland", "A/Norway A/Norway A/Poland")
                text = text.replace("A/CatalA/Brandenburg", "A/Catal A/Brandenburg")
                text = text.replace("A/AlbaniaA/Brandenburg", "A/Albania A/Brandenburg")
                text = text.replace("IVR-237(A/Thailand/08/2022)", "A/Thailand/08/2022")
                text = text.replace("IVR-237", "A/Thailand")
                text = text.replace("9/21NSVH-2067/22/289813/2022 /15/2022 /18/2022 /122/22 /08/2022(A/Thailand/08/22)", "9/21 NSVH161512067/22 289813/2022 15/2022 18/2022 122/22 08/2022 08/22")
                text = text.replace("A/BrandenburgA/MassachusettsA/Thailandassachusetts", "A/Brandenburg A/Massachusetts A/Thailand A/Massachusetts")
                text = text.replace("9/21NSVH-2067/22/289813/2022", "9/21 NSVH161512067/22 289813/2022")
                text = text.replace("NSVH-2067/22/289813/2022", "NSVH161512067/22 289813/2022")
                text = text.replace("A/AlbaniaA/Massachusetts", "A/Albania A/Massachusetts")
                text = text.replace("/10/22 /28719/22 5/21 9/21 NSVH161512067/22 289813/2022 /18/2022 /08/2022 /856/2023 /878/2023", "10/22 28719/22 5/21 9/21 NSVH161512067/22 289813/2022 18/2022 08/2022 856/2023 878/2023")
                text = text.replace("RA/ETFhEuRriEnNgeCnE/1", "A/Thuringen/10/2022")
                text = text.replace("A/ThuringenA/SwitzerlandA/Stockholm", "A/Thuringen A/Switzerland A/Stockholm")
                text = text.replace("/10/22 /28719/22 5/21 9/21 NSVH161512067/22 289813/2022 /18/2022 /08/2022", "10/22 28719/22 5/21 9/21 NSVH161512067/22 289813/2022 18/2022 08/2022")
                text = text.replace("A/Vasteras/SE23-A/Netherlands", "A/Vasteras/SE23- A/Netherlands")
                text = text.replace("/10/22 /28719/22 5/21 9/21SVH-2067/22289813/2022 /18/2022 /08/2022 /856/202310136/RV/2023 14213/2023/10563/2023", "10/22 28719/22 5/21 9/21 NSVH161512067/22 289813/2022 18/2022 08/2022 856/2023 10136/RV/2023 14213/2023 10563/2023")
                text = text.replace("9/21SVH-2067/22/289813/2022", "9/21 NSVH161512067/22 289813/2022")
                text = text.replace("/856/202310136/RV/2023", "856/2023 10136/RV/2023")
                text = text.replace("A/SloveniaA/Switzerland", "A/Slovenia A/Switzerland")
                text = text.replace("/10563/2023", "10563/2023")
                text = text.replace("/49/2024", "49/2024")
                text = text.replace("/8649/2023","8649/2023")
                text = text.replace("A/Netherlands10563/2023", "A/Netherlands/10563/2023")
                text = text.replace("A/Slovenia49/2024", "A/Slovenia/49/2024")
                text = text.replace("A/Switzerland8649/2023", "A/Switzerland/8649/2023")
                text = text.replace("10136/RV/202310136/RV/2023", "10136/RV/2023 10136/RV/2023")
                text = text.replace("/289813/2022 /18/2022 /08/2022 856/2023 10136/RV/2023 10136/RV/2023 10563/2023 49/2024 216/2023IPP29542/2023 /12374/2023/12374/2023", "289813/2022 18/2022 08/2022 856/2023 10136/RV/2023 10136/RV/2023 10563/2023 49/2024 216/2023 IDF-IPP29542/2023 12374/2023 12374/2023")
                text = text.replace("A/France/IDF-", "A/France")
                text = text.replace("/289813/2022 /18/2022 /08/2022 /856/2023 10136/RV/2023 10136/RV/2023 10563/2023 49/2024 216/2023IPP29542/2023 /12374/2023", "289813/2022 18/2022 08/2022 856/2023 10136/RV/2023 10136/RV/2023 10563/2023 49/2024 216/2023 IDF-IPP29542/2023 12374/2023")
                text = text.replace("A/CroatiaA//Netherlands", "A/Croatia A/Netherlands")
                text = text.replace("A/Idaho/A/Oklahoma", "A/Idaho A/Oklahoma")
                text = text.replace("A/Albaniaassachusetts", "A/Albania A/Massachusetts")
                text = text.replace("A/Croatia/A/Netherlands", "A/Croatia A/Netherlands")
                text = text.replace("A/Norway/BurkinaFaso", "A/Norway A/BurkinaFaso")
                text = text.replace("/289813/2022 /18/2022 /08/2022 /856/202310136RV/202310136RV/2023 10563/2023 216/2023 49/2024 /12374/2023 /3131/2023", "289813/2022 18/2022 08/2022 856/2023 10136RV/2023 10136RV/2023 10563/2023 216/2023 49/2024 12374/2023 3131/2023")
                text = text.replace("/18/2022 /08/2022olumbia/27/2023 10136RV/2023 10563/2023 216/2023 49/2024 /47775/2024 /12374/2023 /3131/2023 /423/2023 /1952/2024olumbia/27/2023", "18/2022 08/2022 27/2023 10136RV/2023 10563/2023 216/2023 49/2024 47775/2024 12374/2023 3131/2023 423/2023 1952/2024 27/2023")
                text = text.replace("/18/2022 /08/2022Columbia/27/2023 10136RV/2023 10563/2023 216/2023 49/2024 /47775/2024 /12374/2023 /3131/2023", "18/2022 08/2022 27/2023 10136RV/2023 10563/2023 216/2023 49/2024 47775/2024 12374/2023 3131/2023")
                text = text.replace("/18/2022 /08/2022olumbia/27/2023 10136RV/2023 10563/2023 216/2023 49/2024 /47775/2024 /12374/2023 /3131/2023 10136RV/2023", "18/2022 08/2022 27/2023 10136RV/2023 10563/2023 216/2023 49/2024 47775/2024 12374/2023 3131/2023 10136RV/2023")
                text = text.replace("Starting 1 in 10 dilution", "")
                text = text.replace("A/SwitzerlandA/NetherlandsA/Netherlands", "A/Switzerland A/Netherlands A/Netherlands")
                text = text.replace("A/Croatia/A/DistrictOf", "A/Croatia A/DistrictOf")
                text = text.replace("/05/2024 69/2023 /423/2024 /1952/2024 49/2024 /47775/2024 01285/2024 01285/202410136RV/2023mbia/27/2023", "05/2024 69/2023 423/2024 1952/2024 49/2024 47775/2024 01285/2024 01285/2024 10136RV/2023 27/2023")
                text = text.replace("A/Lyon/CHU/0241955101/2024", "A/Lyon CHU/0241955101/2024")

                lines = text.split('\n')

                tokens = []
                datelist = []

                for line in lines:
                    if line.startswith("Viruses Collection Passage plaques "):
                        tokens = line.split("Viruses Collection Passage plaques ")[1].split(" ")
                    elif line.startswith("Viruses Collection Passage"):
                        tokens = line.split("Viruses Collection Passage ")[1].split(" ")
                    elif line.startswith("Viruses Other Collection Passage"):
                        tokens = line.split("Viruses Other Collection Passage ")[1].split(" ")
                    elif line.startswith("Collection Passage"):
                        tokens = line.split("Collection Passage ")[1].split(" ")
                    elif line.startswith("Date History"):
                        datelist = line.split("Date History ")[1].split(" ")
                    elif line.startswith("information date history"):
                        datelist = line.split("information date history ")[1].split(" ")
                    elif line.startswith("date history"):
                        datelist = line.split("date history ")[1].split(" ")
                    elif line.startswith("Viruses Other"):
                        tokens = line.split("Viruses Other ")[1].split(" ")
                    elif line.startswith("information"):
                        datelist = line.split("information ")[1].split(" ")
                
                pattern = r'^A/.+/$'
                tokens = [s[:-1] if re.match(pattern, s) else s for s in tokens]

                if filename == '2024-feb.pdf':
                    if page_number == 46:
                        tokens = ['A/Thuringen', 'A/Switzerland', 'A/Stockholm', 'A/Darwin', 'A/Catal', 'A/Albania', 'A/Massachusetts', 'A/Thailand', 'A/Sydney', 'A/Sydney']
                        datelist = ['10/22', '28719/22', '5/21', '9/21', 'NSVH161512067/22', '289813/2022', '18/2022', '08/2022', '856/2023', '878/2023']
                
                if filename== '2024-sep.pdf' and page_number >= 56:
                    tokens = []
                    datelist = []


                assert len(tokens) == len(datelist)

                virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')
                full_virus_names = []

                for line in lines:
                    matches = virus_pattern.findall(line)
                    for m in matches:
                        if m not in full_virus_names:
                            full_virus_names.append(m)

                full_virus_names.append("A/Croatia/10136/RV/2023")
                full_virus_names.append("A/France/IDF-IPP29542/2023")
                full_virus_names.append("A/Shanghai-Fengxian/1912/2023")

                col_names = []

                for i, t in enumerate(tokens):
                    matches = [full for full in full_virus_names if full.startswith(t)]

                    if matches:
                        if len(matches) > 1:
                            tid = datelist[i].split("/")[0]
                            for vname in matches:
                                if tid in vname.split("/")[2]:
                                    col_names.append(vname)
                                    break
                        else:
                            col_names.append(matches[0])
                    else:
                        if '/' in t:
                            cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                            if cityname:
                                matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                                if len(matches) > 1:
                                    tid = datelist[i].split("/")[0]
                                    for vname in matches:
                                        if tid in vname.split("/")[2]:
                                            col_names.append(vname)
                                            break
                                else:
                                    col_names.append(matches[0])


                if filename == '2024-sep.pdf':
                    if page_number == 51:
                        col_names = ['A/Albania/289813/2022', 'A/Massachusetts/18/2022',
                                    'A/Thailand/08/2022', 'A/Sydney/856/2023', 'A/Croatia/10136/RV/2023',
                                    'A/Croatia/10136/RV/2023', 'A/Netherlands/10563/2023',
                                    'A/Slovenia/49/2024', 'A/Lisboa/216/2023']
                    elif page_number == 56 or page_number == 57 or page_number == 58:
                        col_names = ['A/Albania/289813/2022', 'A/Massachusetts/18/2022',
                                    'A/Thailand/08/2022', 'A/Sydney/856/2023', 'A/Croatia/10136/RV/2023',
                                    'A/Croatia/10136/RV/2023', 'A/Netherlands/10563/2023',
                                    'A/Slovenia/49/2024', 'A/Lisboa/216/2023', 'A/France/IDF-IPP29542/2023',
                                    'A/Norway/12374/2023']
                    elif page_number == 59 or page_number == 60 or page_number == 61:
                        col_names = ['A/Albania/289813/2022', 'A/Massachusetts/18/2022',
                                    'A/Thailand/08/2022', 'A/Sydney/856/2023', 'A/Croatia/10136/RV/2023',
                                    'A/Croatia/10136/RV/2023', 'A/Netherlands/10563/2023',
                                    'A/Slovenia/49/2024', 'A/Lisboa/216/2023', 'A/France/IDF-IPP29542/2023',
                                    'A/Norway/12374/2023', 'A/BurkinaFaso/3131/2023']
                    elif page_number == 62:
                        col_names = ['A/Croatia/10136/RV/2023',
                                    'A/Croatia/10136/RV/2023',
                                    'A/Netherlands/10563/2023',
                                    'A/Lisboa/216/2023',
                                    'A/Slovenia/49/2024',
                                    'A/District Of Columbia/27/2023',
                                    'A/District Of Columbia/27/2023',
                                    'A/Colorado/06/2024',
                                    'A/Nevada/32/2023',
                                    'A/Idaho/69/2023',
                                    'A/Oklahoma/05/2024']








                data = []
                for line in lines:
                    virus_match = virus_pattern.search(line)
                    if virus_match:
                        virus_name = virus_match.group()

                        def replacer(match):
                            a, b = int(match.group(1)), int(match.group(2))
                            if a % 10 == 0 and b % 10 == 0:
                                avg = (a + b) // 2
                                return str(avg)
                            else:
                                return match.group(0)

                        pattern = re.compile(r'(\d+)-(\d+)')
                        elem = pattern.sub(replacer, line).split(" ")

                        tmp = [s for s in elem if re.fullmatch(r'[0-9<\*]+', s)]
                        elem = []
                        for e in tmp:
                            if "<" in e or "*" in e:
                                elem.append(e)
                            elif int(e) % 10 == 0 and int(e) < 10000:
                                elem.append(e)

                        elem = elem[:len(col_names)]

                        if len(elem) == len(col_names):
                            data.append([virus_name] + elem)
                
                if len(data) > 0:
                    df = pd.DataFrame(data)
                    df.columns = ['Test Virus'] + col_names


                    long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                    virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)
elif filename in  ['2018-feb.pdf', '2019-feb.pdf']:
    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            atext = page.extract_text()
            page_number = pdf.pages.index(page)
            if atext and "Table" in atext and "Neutralisation titre" not in atext and "Neutralisation" not in atext and ("Antigenic analysis of influenza AH3N2 viruses" in atext or "Antigenic analysis of influenza A(H3N2) viruses" in atext or  "Antigenic analyses of influenza A H3N2 viruses" in atext or "Antigenic analysis of influenza A H3N2 viruses" in atext or "Antigenic analyses of influenza A(H3N2) viruses" in atext):
                
                tlist = atext.split("Table ")
                for text in tlist:
                    
                    if len(text) < 10:
                        continue
                
                
                    text = text.replace("15/0934430/09", "15/09 34430/09")
                    text = text.replace(" Serbia ", " A/Serbia ")
                    text = text.replace(" Viruses Collection Passage", "\nViruses Collection Passage")
                    text = text.replace(">", "")
                    text = text.replace("<", "<40")
                    text = text.replace("X-199", "A/Rh")
                    text = text.replace("NIB-85 (A/Almaty/2958/2013)", "A/Almaty/2958/2013")
                    text = text.replace("NIB-85", "A/Alma")
                    text = text.replace("A/PerthA/Alaska", "A/Perth A/Alaska")
                    text = text.replace("A/Alabama5/2010", "A/Alabama/5/2010")
                    text = text.replace("*A/Sth Afr", "A/Sou")
                    text = text.replace("A/Sth Africa", "A/Sou")
                    text = text.replace("A/Sth Afr", "A/Sou")
                    text = text.replace("A/S Africa", "A/Sou")
                    text = text.replace("*A/Stock", "A/Stock")
                    text = text.replace("*A/Nor", "A/Nor")
                    text = text.replace("*A/Switz", "A/Switz")
                    text = text.replace("*", "")
                    text = text.replace("ND", "*")
                    text = text.replace("NT", "*")
                    text = text.replace("2A/HK", "A/HK")
                    text = text.replace("2A/Eng", "A/Eng")
                    text = text.replace("A/Nth Carol", "A/North")
                    text = text.replace(" cl123", "")
                    text = text.replace(" cl121", "")
                    text = text.replace("NYMC X-263B (A/HK/4801/2014)", "A/Hong Kong/4801/2014)")
                    text = text.replace("NYMC X-261 (A/HK/7127/2014)", "A/Hong Kong/7127/2014")
                    text = text.replace("X-263B", "A/Hong")
                    text = text.replace("X-261", "A/Hong")
                    text = text.replace("HK/4801/14", "4801/14")
                    text = text.replace("HK/7127/14", "7127/14")
                    text = text.replace("NYMC X-263B (A/Hong Kong/4801/2014)", "A/Hong Kong/4801/2014")
                    text = text.replace("NYMC X-261 (A/Hong Kong/7127/2014)", "A/Hong Kong/7127/2014")
                    text = text.replace("NIB-93 (A/Hong Kong/7127/2014)", "A/Hong Kong/7127/2014")
                    text = text.replace(" Georgia", " A/Georgia")
                    text = text.replace("plaq 20", "plaq")
                    text = text.replace("A/Georgia HA1 substitutions", "A/Georgia\n HA1 substitutions")
                    text = text.replace("532/15 Egg adaptation", "532/15\n Egg adaptation")
                    text = text.replace("532/15 Kong/4801/2014", "532/15\n Kong/4801/2014")
                    text = text.replace("532/15 A/Hong Kong/4801/2014", "532/15\n A/Hong Kong/4801/2014")
                    text = text.replace("CBER08", "A/Nor")
                    text = text.replace("C1.4", "3806/16")
                    text = text.replace("NIB-103 (A/Norway/3806/2016)", "A/Norway/3806/2016")
                    text = text.replace("NIB-103", "A/Norway/3806/2016")
                    text = text.replace("NYMC X-327 (A/Kansas/14/17)", "A/Kansas/14/2017")
                    text = text.replace("NYMC X-327", "A/Kans")
                    text = text.replace("A/Kans/14/17", "14/17")
                    text = text.replace("A/C'church", "A/Christ")
                    text = text.replace("IVR-197  (A/South Australia/34/2019)", "A/South Australia/34/2019")
                    text = text.replace("IVR-197", "A/South")
                    text = text.replace("A/Sth Aus/34/19", "34/19")
                    text = text.replace("A/Sth Aus", "A/South")
                    text = text.replace("A/DarwinA/NorwayA/NorwayA/Poland","A/Darwin A/Norway A/Norway A/Poland")
                    text = text.replace("A/DarwinA/NorwayA/Norway", "A/Darwin A/Norway A/Norway")
                    text = text.replace("A/NorwayA/NorwayA/Poland", "A/Norway A/Norway A/Poland")
                    text = text.replace("A/CatalA/Brandenburg", "A/Catal A/Brandenburg")
                    text = text.replace("A/AlbaniaA/Brandenburg", "A/Albania A/Brandenburg")
                    text = text.replace("IVR-237(A/Thailand/08/2022)", "A/Thailand/08/2022")
                    text = text.replace("IVR-237", "A/Thailand")
                    text = text.replace("9/21NSVH-2067/22/289813/2022 /15/2022 /18/2022 /122/22 /08/2022(A/Thailand/08/22)", "9/21 NSVH161512067/22 289813/2022 15/2022 18/2022 122/22 08/2022 08/22")
                    text = text.replace("A/BrandenburgA/MassachusettsA/Thailandassachusetts", "A/Brandenburg A/Massachusetts A/Thailand A/Massachusetts")
                    text = text.replace("9/21NSVH-2067/22/289813/2022", "9/21 NSVH161512067/22 289813/2022")
                    text = text.replace("NSVH-2067/22/289813/2022", "NSVH161512067/22 289813/2022")
                    text = text.replace("A/AlbaniaA/Massachusetts", "A/Albania A/Massachusetts")
                    text = text.replace("/10/22 /28719/22 5/21 9/21 NSVH161512067/22 289813/2022 /18/2022 /08/2022 /856/2023 /878/2023", "10/22 28719/22 5/21 9/21 NSVH161512067/22 289813/2022 18/2022 08/2022 856/2023 878/2023")
                    text = text.replace("RA/ETFhEuRriEnNgeCnE/1", "A/Thuringen/10/2022")
                    text = text.replace("A/ThuringenA/SwitzerlandA/Stockholm", "A/Thuringen A/Switzerland A/Stockholm")
                    text = text.replace("/10/22 /28719/22 5/21 9/21 NSVH161512067/22 289813/2022 /18/2022 /08/2022", "10/22 28719/22 5/21 9/21 NSVH161512067/22 289813/2022 18/2022 08/2022")
                    text = text.replace("A/Vasteras/SE23-A/Netherlands", "A/Vasteras/SE23- A/Netherlands")
                    text = text.replace("/10/22 /28719/22 5/21 9/21SVH-2067/22289813/2022 /18/2022 /08/2022 /856/202310136/RV/2023 14213/2023/10563/2023", "10/22 28719/22 5/21 9/21 NSVH161512067/22 289813/2022 18/2022 08/2022 856/2023 10136/RV/2023 14213/2023 10563/2023")
                    text = text.replace("9/21SVH-2067/22/289813/2022", "9/21 NSVH161512067/22 289813/2022")
                    text = text.replace("/856/202310136/RV/2023", "856/2023 10136/RV/2023")
                    text = text.replace("A/SloveniaA/Switzerland", "A/Slovenia A/Switzerland")
                    text = text.replace("/10563/2023", "10563/2023")
                    text = text.replace("/49/2024", "49/2024")
                    text = text.replace("/8649/2023","8649/2023")
                    text = text.replace("A/Netherlands10563/2023", "A/Netherlands/10563/2023")
                    text = text.replace("A/Slovenia49/2024", "A/Slovenia/49/2024")
                    text = text.replace("A/Switzerland8649/2023", "A/Switzerland/8649/2023")
                    text = text.replace("10136/RV/202310136/RV/2023", "10136/RV/2023 10136/RV/2023")
                    text = text.replace("/289813/2022 /18/2022 /08/2022 856/2023 10136/RV/2023 10136/RV/2023 10563/2023 49/2024 216/2023IPP29542/2023 /12374/2023/12374/2023", "289813/2022 18/2022 08/2022 856/2023 10136/RV/2023 10136/RV/2023 10563/2023 49/2024 216/2023 IDF-IPP29542/2023 12374/2023 12374/2023")
                    text = text.replace("A/France/IDF-", "A/France")
                    text = text.replace("/289813/2022 /18/2022 /08/2022 /856/2023 10136/RV/2023 10136/RV/2023 10563/2023 49/2024 216/2023IPP29542/2023 /12374/2023", "289813/2022 18/2022 08/2022 856/2023 10136/RV/2023 10136/RV/2023 10563/2023 49/2024 216/2023 IDF-IPP29542/2023 12374/2023")
                    text = text.replace("A/CroatiaA//Netherlands", "A/Croatia A/Netherlands")
                    text = text.replace("A/Idaho/A/Oklahoma", "A/Idaho A/Oklahoma")
                    text = text.replace("A/Albaniaassachusetts", "A/Albania A/Massachusetts")
                    text = text.replace("A/Croatia/A/Netherlands", "A/Croatia A/Netherlands")
                    text = text.replace("A/Norway/BurkinaFaso", "A/Norway A/BurkinaFaso")
                    text = text.replace("/289813/2022 /18/2022 /08/2022 /856/202310136RV/202310136RV/2023 10563/2023 216/2023 49/2024 /12374/2023 /3131/2023", "289813/2022 18/2022 08/2022 856/2023 10136RV/2023 10136RV/2023 10563/2023 216/2023 49/2024 12374/2023 3131/2023")
                    text = text.replace("/18/2022 /08/2022olumbia/27/2023 10136RV/2023 10563/2023 216/2023 49/2024 /47775/2024 /12374/2023 /3131/2023 /423/2023 /1952/2024olumbia/27/2023", "18/2022 08/2022 27/2023 10136RV/2023 10563/2023 216/2023 49/2024 47775/2024 12374/2023 3131/2023 423/2023 1952/2024 27/2023")
                    text = text.replace("/18/2022 /08/2022Columbia/27/2023 10136RV/2023 10563/2023 216/2023 49/2024 /47775/2024 /12374/2023 /3131/2023", "18/2022 08/2022 27/2023 10136RV/2023 10563/2023 216/2023 49/2024 47775/2024 12374/2023 3131/2023")
                    text = text.replace("/18/2022 /08/2022olumbia/27/2023 10136RV/2023 10563/2023 216/2023 49/2024 /47775/2024 /12374/2023 /3131/2023 10136RV/2023", "18/2022 08/2022 27/2023 10136RV/2023 10563/2023 216/2023 49/2024 47775/2024 12374/2023 3131/2023 10136RV/2023")
                    text = text.replace("Starting 1 in 10 dilution", "")
                    text = text.replace("A/SwitzerlandA/NetherlandsA/Netherlands", "A/Switzerland A/Netherlands A/Netherlands")
                    text = text.replace("A/Croatia/A/DistrictOf", "A/Croatia A/DistrictOf")
                    text = text.replace("/05/2024 69/2023 /423/2024 /1952/2024 49/2024 /47775/2024 01285/2024 01285/202410136RV/2023mbia/27/2023", "05/2024 69/2023 423/2024 1952/2024 49/2024 47775/2024 01285/2024 01285/2024 10136RV/2023 27/2023")
                    text = text.replace("A/Lyon/CHU/0241955101/2024", "A/Lyon CHU/0241955101/2024")
                    text = text.replace("NIB-103 (A/Norway/3806/2016)", "A/Norway/3806/2016")
                    text = text.replace("NYMC X-295 (A/Norway/3806/2016)", "A/Norway/3806/2016")
                    text = text.replace("NYMC X-297 (A/Hong Kong/50/2016)", "A/Hong Kong/50/2016")
                    text = text.replace("NYMC X-303 (hy A/Greece/4/2017)", "A/Greece/4/2017")
                    text = text.replace("NYMC X-303A (hy A/Greece/4/2017)", "A/Greece/4/2017")
                    text = text.replace("NIB 104 (A/Singapore/INFIMH-16-0019/2016)", "A/Singapore/INFIMH-16-0019/2016")
                    text = text.replace("IVR 186 (A/Singapore/INFIMH-16-0019/2016)", "A/Singapore/INFIMH-16-0019/2016")
                    text = text.replace("A/La Rioja/2202/2018", "sojxeosfusofwoeij")
                    text = text.replace("A/La Rioja", "A/La")
                    text = text.replace("sojxeosfusofwoeij", "A/La Rioja/2202/2018")











                    lines = text.split('\n')

                    tokens = []
                    datelist = []

                    for line in lines:
                        if line.startswith("Viruses Collection Passage plaques "):
                            tokens = line.split("Viruses Collection Passage plaques ")[1].split(" ")
                        elif line.startswith("Viruses Collection Passage"):
                            tokens = line.split("Viruses Collection Passage ")[1].split(" ")
                        elif line.startswith("Viruses Other Collection Passage"):
                            tokens = line.split("Viruses Other Collection Passage ")[1].split(" ")
                        elif line.startswith("Collection Passage"):
                            tokens = line.split("Collection Passage ")[1].split(" ")
                        elif line.startswith("Date History"):
                            datelist = line.split("Date History ")[1].split(" ")
                        elif line.startswith("information date history"):
                            datelist = line.split("information date history ")[1].split(" ")
                        elif line.startswith("date history"):
                            datelist = line.split("date history ")[1].split(" ")
                        elif line.startswith("Viruses Other"):
                            tokens = line.split("Viruses Other ")[1].split(" ")
                        elif line.startswith("information"):
                            datelist = line.split("information ")[1].split(" ")
                    
                    pattern = r'^A/.+/$'
                    tokens = [s[:-1] if re.match(pattern, s) else s for s in tokens]

                    if filename == '2024-feb.pdf':
                        if page_number == 46:
                            tokens = ['A/Thuringen', 'A/Switzerland', 'A/Stockholm', 'A/Darwin', 'A/Catal', 'A/Albania', 'A/Massachusetts', 'A/Thailand', 'A/Sydney', 'A/Sydney']
                            datelist = ['10/22', '28719/22', '5/21', '9/21', 'NSVH161512067/22', '289813/2022', '18/2022', '08/2022', '856/2023', '878/2023']
                    
                    if filename== '2024-sep.pdf' and page_number >= 56:
                        tokens = []
                        datelist = []

                    if filename == '2018-feb.pdf':
                        if "8-5." in text:
                            tokens = []
                            datelist = []

                    assert len(tokens) == len(datelist)

                    virus_pattern = re.compile(r'A/[^/\n]+/[A-Za-z0-9\-]+/\d{2,4}')
                    full_virus_names = []

                    for line in lines:
                        matches = virus_pattern.findall(line)
                        for m in matches:
                            if m not in full_virus_names:
                                full_virus_names.append(m)

                    full_virus_names.append("A/Oman/2585/2016")
                    full_virus_names.append("A/Norway/4436/2016")
                    full_virus_names.append("A/Greece/4/2017")
                    full_virus_names.append("A/Hong Kong/4018/2017")
                    full_virus_names.append("A/Norway/4293/2017")
                    full_virus_names.append("A/La Rioja/2202/2018")
                    full_virus_names.append("A/Norway/2516/2018")
                    full_virus_names.append("A/Hong Kong/675/2018")
                    full_virus_names.append("A/Norway/3275/18")

                    col_names = []

                    for i, t in enumerate(tokens):
                        matches = [full for full in full_virus_names if full.startswith(t)]

                        if matches:
                            if len(matches) > 1:
                                tid = datelist[i].split("/")[0]
                                for vname in matches:
                                    if tid in vname.split("/")[2]:
                                        col_names.append(vname)
                                        break
                            else:
                                col_names.append(matches[0])
                        else:
                            if '/' in t:
                                cityname = CITY_ABBREVIATION_DICT.get(t.split('/')[1], None)
                                if cityname:
                                    matches = [full for full in full_virus_names if full.startswith('A/'+cityname)]
                                    if len(matches) > 1:
                                        tid = datelist[i].split("/")[0]
                                        for vname in matches:
                                            if tid in vname.split("/")[2]:
                                                col_names.append(vname)
                                                break
                                    else:
                                        col_names.append(matches[0])




                    if filename == '2018-feb.pdf':
                        if "8-5." in text:
                            col_names = ['A/Hong Kong/4801/2014', 'A/Norway/3806/2016', 'A/Norway/3806/2016', 
                                        'A/Hong Kong/50/2016', 'A/Hong Kong/50/2016', 'A/Greece/4/2017', 'A/Greece/4/2017', 
                                        'A/Singapore/INFIMH-16-0019/2016', 'A/Singapore/INFIMH-16-0019/2016', 
                                        'A/Singapore/INFIMH-16-0019/2016']
                                





                    data = []
                    for line in lines:
                        virus_match = virus_pattern.search(line)
                        if virus_match:
                            virus_name = virus_match.group()

                            def replacer(match):
                                a, b = int(match.group(1)), int(match.group(2))
                                if a % 10 == 0 and b % 10 == 0:
                                    avg = (a + b) // 2
                                    return str(avg)
                                else:
                                    return match.group(0)

                            pattern = re.compile(r'(\d+)-(\d+)')
                            elem = pattern.sub(replacer, line).split(" ")

                            tmp = [s for s in elem if re.fullmatch(r'[0-9<\*]+', s)]
                            elem = []
                            for e in tmp:
                                if "<" in e or "*" in e:
                                    elem.append(e)
                                elif int(e) % 10 == 0 and int(e) < 10000:
                                    elem.append(e)

                            elem = elem[:len(col_names)]

                            if len(elem) == len(col_names):
                                data.append([virus_name] + elem)
                    
                    if len(data) > 0:
                        df = pd.DataFrame(data)
                        df.columns = ['Test Virus'] + col_names


                        long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                        virus_df = pd.concat([virus_df, long_df], ignore_index=True)


    virus_df.to_csv(out_folder + filename.replace(".pdf", "") + ".csv", index=False)



