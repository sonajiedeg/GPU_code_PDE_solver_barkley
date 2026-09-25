void solSave(char* fname, std::vector<float> sol, SimParamStruct params){
	const unsigned int s = sol.size();
	std::fstream newfile;
	newfile.open(fname, ios::out);
    if (newfile.is_open())
    {
        newfile << scientific << setprecision(17);
        for (int i = 0; i < s; i++) { newfile << sol[i] << "\n"; }
        newfile.close();
    }
}
